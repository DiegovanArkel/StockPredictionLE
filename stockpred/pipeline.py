"""End-to-end orchestration: fetch -> features -> walk-forward -> CQR ->
GARCH -> production refit -> assemble -> backtest -> write_artifacts
(Task 10).

``run`` is the single entry point the CLI (``stockpred.__main__``) and the
GitHub Actions weekly cron both call. It stitches together every module
built in Tasks 1-9 without adding any new modeling logic of its own.

Production forecast (full-sample MacroFactorExtractor)
--------------------------------------------------------
Every *reported metric* (``wf.metrics``, the backtest) comes from
:func:`stockpred.models.workhorse.run_walk_forward`, which fits
``MacroFactorExtractor`` on train-fold months only -- never on the full
sample -- so nothing here inflates those numbers with look-ahead.

The one exception, by design, is the *production* forecast written to
``forecasts.parquet``: since there is no future to hold out for the
forecast actually being shipped, ``MacroFactorExtractor`` is fit on the
FULL macro history for that step only (see ``_production_forecast``). This
mirrors the walk-forward harness's own fit/cal/oos split (Task 5/7) at the
scale of the whole sample: a trailing ``CAL_MONTHS`` calibration block
still gives ``cqr_offsets`` a genuinely held-out calibration set for the
production forecast's own CQR offsets, even though the macro factors
themselves are fit on everything.

Every production forecast row also carries the ``pred_date`` it was
actually predicted from (the panel date its features came from, not
today's date), and a ticker whose latest panel month is stale relative to
the rest of the universe (delisted, or its feed silently stopped
updating) is excluded from the forecast entirely rather than shipped
looking as current as everything else -- see ``_production_forecast``.
"""

from __future__ import annotations

import logging
import time
from datetime import date as _date
from pathlib import Path

import pandas as pd

from stockpred.artifacts import assemble_forecasts, write_artifacts
from stockpred.backtest import run_backtest
from stockpred.config import Config, load_config
from stockpred.data.factors import fetch_famafrench
from stockpred.data.macro import fetch_ecb, fetch_fred
from stockpred.data.prices import fetch_prices
from stockpred.data.storage import cached_fetch, load_parquet, refresh_duckdb
from stockpred.features.macro_pca import MacroFactorExtractor, build_macro_wide
from stockpred.features.stock import build_monthly_panel
from stockpred.models.conformal import apply_cqr, calibrate_from_wf, cqr_offsets
from stockpred.models.garch import garch_all
from stockpred.models.workhorse import (
    CAL_MONTHS,
    STOCK_FEATURE_COLS,
    WorkhorseModel,
    run_walk_forward,
)
from stockpred.validation import decay_weights

logger = logging.getLogger(__name__)

_MACRO_COLUMNS = ["date", "series_id", "value"]

# A ticker whose last available panel month is more than this many months
# older than the freshest ticker's last month (e.g. delisted, or its price
# feed silently stopped updating) is excluded from the production forecast
# rather than being predicted from stale, months-old features.
_STALE_MONTHS = 3

# Mirrors stockpred.models.workhorse._MIN_FIT_MONTHS: the minimum number of
# fit-only months required after carving CAL_MONTHS off the training panel,
# so the one-shot production calibration split has the same guard the
# per-fold walk-forward split enforces.
_MIN_FIT_MONTHS = 12


class _StageTimer:
    """Context manager that logs and records elapsed seconds for one stage."""

    def __init__(self, name: str, timings: dict[str, float]):
        self.name = name
        self.timings = timings
        self._start = 0.0

    def __enter__(self) -> "_StageTimer":
        self._start = time.monotonic()
        logger.info("stage %r: starting", self.name)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.monotonic() - self._start
        self.timings[self.name] = elapsed
        if exc_type is None:
            logger.info("stage %r: done in %.2fs", self.name, elapsed)
        else:
            logger.info("stage %r: failed after %.2fs", self.name, elapsed)
        return False


def _fetch_all(cfg: Config) -> tuple[list[str], dict[str, bool]]:
    """Fetch prices/macro/factors via ``cached_fetch`` (falling back to the
    parquet cache on any failure) and report failed tickers + per-source
    staleness."""
    failures: list[str] = []
    staleness: dict[str, bool] = {}

    def _fetch_prices_fn() -> pd.DataFrame:
        df, failed = fetch_prices(cfg.tickers, cfg.price_start, yf_suffix=cfg.yf_suffix)
        failures.extend(failed)
        return df

    _, stale = cached_fetch("prices", _fetch_prices_fn, cfg.data_dir)
    staleness["prices"] = stale

    def _fetch_macro_fn() -> pd.DataFrame:
        # Both fetch_fred and fetch_ecb are best-effort per series already
        # (see stockpred.data.macro) -- combine into the single tidy "macro"
        # dataset cached_fetch stores.
        fred_df = fetch_fred(cfg.fred_series)
        ecb_df = fetch_ecb(cfg.ecb_series)
        return pd.concat([fred_df, ecb_df], ignore_index=True)[_MACRO_COLUMNS]

    _, stale = cached_fetch("macro", _fetch_macro_fn, cfg.data_dir)
    staleness["macro"] = stale

    def _fetch_factors_fn() -> pd.DataFrame:
        return fetch_famafrench(cfg.price_start)

    _, stale = cached_fetch("factors", _fetch_factors_fn, cfg.data_dir)
    staleness["factors"] = stale

    return failures, staleness


def fetch(config_path: str | Path = "config.yaml") -> dict:
    """Fetch and cache prices/macro/factors data only (``stockpred fetch``)."""
    cfg = load_config(config_path)
    failures, staleness = _fetch_all(cfg)
    return {"failures": failures, "staleness": staleness}


def _production_feature_cols(cfg: Config) -> list[str]:
    return STOCK_FEATURE_COLS + [f"F{i + 1}" for i in range(cfg.n_pca_factors)]


def _merge_factors(rows: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    return rows.merge(factors, left_on="date", right_index=True, how="left")


def _production_forecast(
    panel: pd.DataFrame,
    panel_full: pd.DataFrame,
    macro_wide: pd.DataFrame,
    cfg: Config,
    lgb_params: dict | None,
) -> tuple[pd.DataFrame, dict, list[str]]:
    """Refit the workhorse model on ALL available data and predict the
    latest available month per ticker.

    ``MacroFactorExtractor`` is fit on the FULL ``macro_wide`` sample here --
    the one place in the pipeline that's allowed, since this is the
    production forecast itself, not a reported metric (see module
    docstring).

    Tickers whose latest panel month is more than ``_STALE_MONTHS`` months
    older than the freshest ticker's latest month (e.g. delisted, or a
    price feed that silently stopped updating) are excluded from the
    prediction entirely -- their last real feature row is too old to trust
    -- and returned separately as ``stale_tickers`` rather than being
    predicted from months-stale features and stamped with today's
    ``as_of`` date as if they were current.

    Returns ``(final_pred, offsets_prod, stale_tickers)``, where
    ``final_pred`` additionally carries a ``pred_date`` column -- the panel
    date each row's features actually came from -- for
    :func:`stockpred.artifacts.assemble_forecasts` to pass through into the
    written forecast.
    """
    feature_cols = _production_feature_cols(cfg)

    extractor = MacroFactorExtractor(cfg.n_pca_factors).fit(macro_wide)
    factors_wide = extractor.transform(macro_wide)

    train_rows = _merge_factors(panel, factors_wide)

    unique_months = pd.DatetimeIndex(sorted(train_rows["date"].unique()))
    if len(unique_months) < CAL_MONTHS + _MIN_FIT_MONTHS:
        raise ValueError(
            f"_production_forecast: only {len(unique_months)} train months "
            f"available, need at least {CAL_MONTHS + _MIN_FIT_MONTHS} "
            f"({CAL_MONTHS} calibration + {_MIN_FIT_MONTHS} minimum fit)"
        )

    latest_per_ticker = (
        panel_full.sort_values("date").groupby("ticker", as_index=False, sort=False).tail(1)
    )
    global_max_date = latest_per_ticker["date"].max()
    stale_cutoff = global_max_date - pd.DateOffset(months=_STALE_MONTHS)
    is_fresh = latest_per_ticker["date"] >= stale_cutoff
    stale_tickers = sorted(latest_per_ticker.loc[~is_fresh, "ticker"].tolist())

    pred_rows = _merge_factors(latest_per_ticker.loc[is_fresh], factors_wide)

    cal_months = unique_months[-CAL_MONTHS:]
    fit_months = unique_months[:-CAL_MONTHS]

    fit_rows = train_rows[train_rows["date"].isin(fit_months)]
    cal_rows = train_rows[train_rows["date"].isin(cal_months)]

    # --- Stage A: fit-only model -> calibration predictions -> offsets ---
    X_fit = fit_rows[feature_cols]
    y_fit = fit_rows["fwd_ret_1m"]
    w_fit = decay_weights(fit_rows["date"], cfg.decay_half_life_years)

    model_fit = WorkhorseModel(cfg.quantiles, lgb_params=lgb_params)
    model_fit.fit(X_fit, y_fit, sample_weight=w_fit)

    cal_preds = model_fit.predict(cal_rows[feature_cols])
    offset_90 = cqr_offsets(cal_preds["q05"], cal_preds["q95"], cal_rows["fwd_ret_1m"], alpha=0.10)
    offset_50 = cqr_offsets(cal_preds["q25"], cal_preds["q75"], cal_rows["fwd_ret_1m"], alpha=0.50)
    offsets_prod = {"90": offset_90, "50": offset_50}

    # --- Stage B: refit on ALL train rows -> predict the latest month ---
    X_all = train_rows[feature_cols]
    y_all = train_rows["fwd_ret_1m"]
    w_all = decay_weights(train_rows["date"], cfg.decay_half_life_years)

    model_all = WorkhorseModel(cfg.quantiles, lgb_params=lgb_params)
    model_all.fit(X_all, y_all, sample_weight=w_all)

    preds = model_all.predict(pred_rows[feature_cols])
    final_pred = pd.DataFrame(
        {
            "ticker": pred_rows["ticker"].to_numpy(),
            "pred_date": pred_rows["date"].to_numpy(),
        }
    )
    for col in preds.columns:
        final_pred[col] = preds[col].to_numpy()

    final_pred = apply_cqr(final_pred, offsets_prod)
    return final_pred, offsets_prod, stale_tickers


def run(
    config_path: str | Path = "config.yaml",
    skip_fetch: bool = False,
    lgb_params: dict | None = None,
) -> dict:
    """Run the full pipeline: fetch -> duckdb refresh -> features ->
    walk-forward -> CQR -> backtest -> GARCH -> production refit ->
    assemble -> write_artifacts.

    Returns a summary dict ``{"timings": {...}, "n_forecasts": int,
    "metrics_keys": [...]}``.
    """
    cfg = load_config(config_path)
    timings: dict[str, float] = {}
    failures: list[str] = []
    staleness: dict[str, bool] = {}

    if skip_fetch:
        logger.info("stage 'fetch': skipped (--skip-fetch)")
    else:
        with _StageTimer("fetch", timings):
            failures, staleness = _fetch_all(cfg)

    with _StageTimer("refresh_duckdb", timings):
        refresh_duckdb(cfg.duckdb_path, cfg.data_dir)

    with _StageTimer("load_data", timings):
        prices_df = load_parquet("prices", cfg.data_dir)
        factors_df = load_parquet("factors", cfg.data_dir)
        macro_df = load_parquet("macro", cfg.data_dir)
        if prices_df is None or factors_df is None:
            raise RuntimeError(
                "pipeline.run: prices/factors parquet cache missing under "
                f"{cfg.data_dir} -- run without --skip-fetch first"
            )
        if macro_df is None:
            macro_df = pd.DataFrame(columns=_MACRO_COLUMNS)

    with _StageTimer("features", timings):
        panel = build_monthly_panel(prices_df, factors_df)
        panel_full = build_monthly_panel(prices_df, factors_df, drop_missing_target=False)
        # Clip macro history to price_start minus a 2-year buffer (room for
        # the 12-month log-diff lookback) before computing the wide panel.
        # FRED's keyless endpoint returns each series' FULL history (e.g.
        # INDPRO back to 1919); with no clip that forces build_macro_wide's
        # missing-fraction column drop to evaluate every series against an
        # 1919-2026 index, which drops later-starting series (VIXCLS,
        # DTWEXBGS, ...) as "mostly missing" even though they're fully
        # populated over the window the model actually trains on -- see
        # stockpred.features.macro_pca.build_macro_wide's min_date docstring.
        macro_min_date = pd.Timestamp(cfg.price_start) - pd.DateOffset(years=2)
        macro_wide = build_macro_wide(macro_df, min_date=macro_min_date)

    with _StageTimer("walk_forward", timings):
        wf = run_walk_forward(panel, macro_wide, cfg, lgb_params=lgb_params)

    with _StageTimer("calibration", timings):
        offsets, oos_calibrated = calibrate_from_wf(wf)

    with _StageTimer("backtest", timings):
        backtest_summary = run_backtest(oos_calibrated, cfg)

    with _StageTimer("garch", timings):
        garch_df = garch_all(prices_df)

    with _StageTimer("production_forecast", timings):
        final_pred, offsets_prod, stale_tickers = _production_forecast(
            panel, panel_full, macro_wide, cfg, lgb_params
        )
        if stale_tickers:
            logger.warning(
                "excluding %d stale ticker(s) from the production forecast "
                "(last panel month more than %d months before the freshest "
                "ticker's): %s",
                len(stale_tickers),
                _STALE_MONTHS,
                stale_tickers,
            )
            failures = sorted(set(failures) | set(stale_tickers))
            garch_df = garch_df[~garch_df["ticker"].isin(stale_tickers)].reset_index(drop=True)

    with _StageTimer("assemble_and_write", timings):
        as_of = _date.today().isoformat()
        forecasts = assemble_forecasts(final_pred, garch_df, offsets_prod, as_of=as_of)
        write_artifacts(
            cfg.artifacts_dir,
            forecasts,
            wf.metrics,
            offsets,
            backtest_summary,
            history=oos_calibrated,
            failures=failures,
            staleness=staleness,
        )

    summary = {
        "timings": timings,
        "n_forecasts": int(len(forecasts)),
        "metrics_keys": sorted(wf.metrics.keys()),
    }
    logger.info("pipeline run complete: %s", summary)
    return summary

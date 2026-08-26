"""StockPred Streamlit dashboard (Task 11).

Standalone artifact reader: only ``pandas``/``pyarrow``/``json`` are used to
read ``artifacts/*.parquet|json`` -- this module (and ``charts.py``) must
never import ``stockpred`` or any of its heavy modeling dependencies
(lightgbm, arch, scikit-learn), so a Streamlit Cloud deploy only needs the
lightweight requirements in ``requirements.txt``. See
``stockpred/artifacts.py`` for the authoritative artifact contract this
mirrors (``write_artifacts``/``load_artifacts``).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Make sibling module `charts.py` importable regardless of how this script
# is launched (`streamlit run app/streamlit_app.py`, AppTest, pytest, ...).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import charts  # noqa: E402

CAVEAT = "Educational tool — not financial advice. Expected edge after costs is likely zero."
_STALE_DAYS = 40
_QUANTILE_SUFFIXES = ["05", "25", "50", "75", "95"]
_QUANTILE_KEYS = [f"q{s}" for s in _QUANTILE_SUFFIXES]


def resolve_artifacts_dir() -> Path:
    """Default: ``<repo_root>/artifacts`` next to this file's parent
    directory. Overridable via ``STOCKPRED_ARTIFACTS_DIR``.
    """
    env = os.environ.get("STOCKPRED_ARTIFACTS_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "artifacts"


@st.cache_data(ttl=3600, show_spinner="Loading artifacts...")
def load_artifacts_cached(artifacts_dir_str: str) -> dict[str, Any]:
    """Standalone re-implementation of ``stockpred.artifacts.load_artifacts``
    semantics: a missing file (including a missing directory) maps its key
    to ``None`` rather than raising; a present-but-corrupt file raises.
    """
    artifacts_dir = Path(artifacts_dir_str)

    def _read_parquet(name: str) -> pd.DataFrame | None:
        path = artifacts_dir / name
        return pd.read_parquet(path) if path.exists() else None

    def _read_json(name: str) -> Any:
        path = artifacts_dir / name
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    return {
        "forecasts": _read_parquet("forecasts.parquet"),
        "oos_history": _read_parquet("oos_history.parquet"),
        "diagnostics": _read_json("diagnostics.json"),
        "backtest": _read_json("backtest.json"),
        "manifest": _read_json("manifest.json"),
    }


def render_banner() -> None:
    st.warning(CAVEAT, icon="⚠️")


# --------------------------------------------------------------------------
# Forecasts page
# --------------------------------------------------------------------------

def _quantile_row(row: pd.Series, col_prefix: str) -> dict[str, float]:
    """Read a quantile family off ``row`` given its column prefix (e.g.
    ``"q"`` for ``q05..q95``, ``"ens_q"`` for ``ens_q05..ens_q95``, ``"y_q"``
    for ``y_q05..y_q95``, ``"raw_q"`` for ``raw_q05..raw_q95``), returning a
    dict keyed by the plain ``q05..q95`` names the chart functions expect.
    """
    return {f"q{s}": float(row[f"{col_prefix}{s}"]) for s in _QUANTILE_SUFFIXES}


def _cumulative_index(history: pd.DataFrame, ticker: str, n_months: int = 24) -> tuple[list, list[float]]:
    """Last ``n_months`` of a ticker's OOS ``y_true`` monthly returns,
    cumulated into a return index starting at 100.

    Date convention (off-by-one fix): a history row dated ``t`` carries
    ``y_true(t) = fwd_ret_1m(t)``, the return *realized during month t+1*
    (see ``stockpred.features.stock.build_monthly_panel``). Plotting that
    return at ``t`` would draw every realized move one month too early and
    make the chart's right edge overlap the forecast's own month. Each
    point is therefore plotted at ``t + 1 month`` — the month the return
    actually happened — so the last plotted point sits immediately before
    the forecast fan, with no overlap and no gap.
    """
    sub = history[history["ticker"] == ticker].sort_values("date").tail(n_months)
    if sub.empty:
        return [], []
    idx = 100.0
    dates: list = []
    values: list[float] = []
    for d, ret in zip(sub["date"], sub["y_true"]):
        idx *= 1.0 + float(ret)
        # Shift the *label*, not the value: this return was realized during
        # the month after the row's feature date.
        dates.append(pd.Timestamp(d) + pd.DateOffset(months=1))
        values.append(idx)
    return dates, values


def page_forecasts(data: dict[str, Any]) -> None:
    st.header("Forecasts")

    forecasts = data["forecasts"]
    oos_history = data["oos_history"]

    if forecasts is None or forecasts.empty:
        st.info(
            "No forecasts available yet -- run the pipeline to generate "
            "`artifacts/forecasts.parquet` (see Task 12)."
        )
        return

    tickers = sorted(forecasts["ticker"].unique())
    ticker = st.selectbox("Ticker", tickers)
    ensemble = st.checkbox(
        "Show ensemble (50/50 LightGBM + GARCH average) instead of the calibrated LightGBM quantiles",
        value=False,
    )

    row = forecasts.loc[forecasts["ticker"] == ticker].iloc[0]

    as_of = row.get("as_of")
    pred_date = row.get("pred_date", as_of)
    st.caption(f"As of: {as_of}  |  Forecast panel date: {pred_date}")

    if pred_date is not None and as_of is not None:
        try:
            days_stale = (pd.Timestamp(as_of) - pd.Timestamp(pred_date)).days
            if days_stale > _STALE_DAYS:
                st.error(
                    f"Stale forecast: the underlying panel data for {ticker} is "
                    f"{days_stale} days old (as of {as_of}). Treat this forecast with caution."
                )
        except (TypeError, ValueError):
            pass

    prefix = "ens_q" if ensemble else "q"
    monthly_q = _quantile_row(row, prefix)

    hist_dates, hist_index = ([], [])
    if oos_history is not None and not oos_history.empty:
        hist_dates, hist_index = _cumulative_index(oos_history, ticker)
    has_hist = bool(hist_dates)

    # Same convention as _cumulative_index: the forecast made from panel
    # month `pred_date` is a forecast of the return realized during the
    # FOLLOWING month, so the fan's tip is drawn one month after pred_date.
    forecast_month = (
        pd.Timestamp(pred_date) + pd.DateOffset(months=1) if pred_date is not None else None
    )

    if has_hist:
        last_date = pd.Timestamp(hist_dates[-1])
        fan_date = forecast_month if forecast_month is not None else last_date + pd.DateOffset(months=1)
        if fan_date <= last_date:
            fan_date = last_date + pd.DateOffset(months=1)
    else:
        fan_date = forecast_month if forecast_month is not None else pd.Timestamp.today()
        # No OOS history to anchor the fan's left edge -- without a distinct
        # start point the fan collapses to a single x position and the bands
        # never visibly widen. Anchor ~30 days (one month) before the
        # forecast date instead, at the base-100 index, so the fan still
        # fans out.
        anchor_date = fan_date - pd.Timedelta(days=30)
        hist_dates, hist_index = [anchor_date], [100.0]

    fig = charts.fan_chart(
        hist_dates, hist_index, fan_date, monthly_q,
        title=f"{ticker}: indexed history (last 24m) + next-month fan"
              f"{' (ensemble)' if ensemble else ''}",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "History points are dated by the month the return was *realized* "
        "(one month after the feature row it came from), and the fan tip "
        "sits in the month this forecast is about."
    )
    if not has_hist:
        st.caption("No OOS history for this ticker yet -- fan shown starting from an index of 100.")

    st.subheader("12-month scenario range")
    st.caption("Wide, humble scenario range — not a prediction.")
    y_q = _quantile_row(row, "y_q")
    st.plotly_chart(charts.humble_band_chart(y_q), width="stretch")

    st.subheader("Quantile table")
    raw_q = None
    if not ensemble:
        # There is no raw_q50 column -- only the tails are CQR-calibrated
        # (see stockpred/models/conformal.py apply_cqr), so the raw and
        # calibrated medians are identical by construction.
        raw_q = {
            "q05": float(row["raw_q05"]), "q25": float(row["raw_q25"]),
            "q50": float(row["q50"]),
            "q75": float(row["raw_q75"]), "q95": float(row["raw_q95"]),
        }
    table_rows = {
        "Monthly (1m)": monthly_q,
        "Yearly (12m)": y_q,
    }
    table_df = pd.DataFrame(table_rows).T[_QUANTILE_KEYS]
    st.dataframe(table_df.style.format("{:.1%}"), width="stretch")

    if raw_q is not None:
        with st.expander("Pre-calibration (raw) LightGBM monthly quantiles"):
            raw_df = pd.DataFrame([raw_q], index=["Monthly (raw)"])[_QUANTILE_KEYS]
            st.dataframe(raw_df.style.format("{:.1%}"), width="stretch")

    ann_vol = row.get("ann_vol")
    converged = row.get("garch_converged")
    if ann_vol is not None:
        st.caption(
            f"Annualized volatility (GARCH): {float(ann_vol):.1%}  |  "
            f"GARCH converged: {bool(converged)}"
        )


# --------------------------------------------------------------------------
# Diagnostics page
# --------------------------------------------------------------------------

def page_diagnostics(data: dict[str, Any]) -> None:
    st.header("Diagnostics")

    diagnostics = data["diagnostics"]
    manifest = data["manifest"]

    if diagnostics is None:
        st.info(
            "No diagnostics available yet -- run the pipeline to generate "
            "`artifacts/diagnostics.json` (see Task 12)."
        )
    else:
        wf_metrics = diagnostics.get("wf_metrics", {})
        offsets = diagnostics.get("offsets", {})
        pooled = wf_metrics.get("pooled", {})
        folds = wf_metrics.get("folds", {})

        col1, col2 = st.columns(2)
        with col1:
            if folds:
                fold_r2 = {k: v.get("r2_oos_median") for k, v in folds.items()}
                st.plotly_chart(
                    charts.r2_bar(fold_r2, pooled.get("r2_oos_median")),
                    width="stretch",
                )
            else:
                st.info("No per-fold metrics available.")
        with col2:
            if "coverage_raw_90" in offsets and "coverage_cal_90" in offsets:
                st.plotly_chart(
                    charts.coverage_chart(offsets["coverage_raw_90"], offsets["coverage_cal_90"]),
                    width="stretch",
                )
            else:
                st.info("No coverage data available.")

        st.subheader("Pinball loss (pooled OOS, by quantile)")
        pinball = pooled.get("pinball_loss")
        if pinball:
            st.dataframe(
                pd.DataFrame([pinball]).style.format("{:.4f}"),
                width="stretch",
            )
        else:
            st.info("No pinball loss data available.")

        st.subheader("Feature importance")
        importance = diagnostics.get("feature_importance")
        if importance:
            st.plotly_chart(charts.feature_importance_bar(importance), width="stretch")
        else:
            st.info("No feature importance data available.")

        st.subheader("Conformal calibration offsets")
        if offsets:
            st.write(
                f"Pooled 90% interval offset: `{offsets.get('90', float('nan')):+.1%}`  |  "
                f"pooled 50% interval offset: `{offsets.get('50', float('nan')):+.1%}`"
            )
            st.caption(
                "These pooled offsets are a whole-sample summary only. The "
                "out-of-sample intervals above are calibrated **per fold** — "
                "each fold uses offsets computed from its own, strictly "
                "earlier calibration months — and the shipped forecast uses "
                "its own trailing calibration block."
            )
            per_fold = offsets.get("per_fold")
            if per_fold:
                with st.expander("Per-fold conformal offsets (what actually calibrated the OOS intervals)"):
                    pf_df = pd.DataFrame(
                        [
                            {"fold": int(k), "offset_90": v.get("90"), "offset_50": v.get("50")}
                            for k, v in sorted(per_fold.items(), key=lambda kv: int(kv[0]))
                        ]
                    )
                    st.dataframe(
                        pf_df.style.format({"offset_90": "{:+.2%}", "offset_50": "{:+.2%}"}),
                        width="stretch",
                    )
        else:
            st.info("No offsets available.")

    with st.expander("Run manifest"):
        if manifest:
            st.write(f"Generated at: {manifest.get('generated_at')}")
            st.write(f"Forecasts: {manifest.get('n_forecasts')}  |  History rows: {manifest.get('n_history_rows')}")
            failed = manifest.get("failed_tickers") or []
            st.write(f"Failed tickers ({len(failed)}): {', '.join(failed) if failed else 'none'}")
            stale = manifest.get("stale_sources") or {}
            st.write(f"Stale sources: {stale if stale else 'none'}")
            st.write("Package versions:")
            st.json(manifest.get("versions", {}))
        else:
            st.info("No manifest available yet.")


# --------------------------------------------------------------------------
# Backtest page
# --------------------------------------------------------------------------

def page_backtest(data: dict[str, Any]) -> None:
    st.header("Backtest")

    backtest = data["backtest"]

    if backtest is None:
        st.info(
            "No backtest results available yet -- run the pipeline to generate "
            "`artifacts/backtest.json` (see Task 12)."
        )
        return

    monthly_returns = backtest.get("monthly_returns", [])
    fig = charts.equity_curve(monthly_returns, backtest.get("benchmark_total_return"))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "The benchmark line marks where an equal-weight buy-and-hold portfolio "
        "would have ended up (its compounded total return) -- per-month benchmark "
        "equity isn't recorded, so this is an endpoint marker, not a real path."
    )

    row1 = st.columns(4)
    row1[0].metric("Total return", f"{backtest.get('total_return', 0.0):.1%}")
    row1[1].metric("Annualized return", f"{backtest.get('ann_return', 0.0):.1%}")
    row1[2].metric("Annualized vol", f"{backtest.get('ann_vol', 0.0):.1%}")
    row1[3].metric("Sharpe", f"{backtest.get('sharpe', 0.0):.2f}")

    row2 = st.columns(4)
    row2[0].metric("DSR", f"{backtest.get('dsr', 0.0):.2f}")
    row2[1].metric("PSR", f"{backtest.get('psr', 0.0):.2f}")
    row2[2].metric("Max drawdown", f"{backtest.get('max_drawdown', 0.0):.1%}")
    row2[3].metric("Trades", f"{backtest.get('n_trades', 0)}")

    row3 = st.columns(2)
    row3[0].metric("Final capital (EUR)", f"{backtest.get('final_capital', 0.0):,.0f}")
    row3[1].metric("Months backtested", f"{backtest.get('n_months', 0)}")

    st.caption(
        "DSR (Deflated Sharpe Ratio) corrects the Sharpe ratio for the number of "
        "strategy variants effectively trialed across folds, guarding against "
        "overstating skill from multiple testing."
    )

    with st.expander("Monthly returns"):
        if monthly_returns:
            mr_df = pd.DataFrame(monthly_returns)
            for col in ("ret", "gross_ret"):
                if col in mr_df.columns:
                    mr_df[col] = mr_df[col].map(lambda v: f"{v:.1%}")
            st.dataframe(mr_df, width="stretch")
        else:
            st.info("No monthly returns recorded.")

    per_fold_sharpe = backtest.get("per_fold_sharpe")
    if per_fold_sharpe:
        with st.expander("Per-fold Sharpe"):
            st.dataframe(pd.DataFrame(per_fold_sharpe), width="stretch")


# --------------------------------------------------------------------------
# App entrypoint
# --------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="StockPred Dashboard", layout="wide")
    render_banner()

    artifacts_dir = resolve_artifacts_dir()
    data = load_artifacts_cached(str(artifacts_dir))

    page = st.sidebar.radio("Navigate", ["Forecasts", "Diagnostics", "Backtest"])
    st.sidebar.caption(f"Artifacts dir: `{artifacts_dir}`")
    if data["manifest"]:
        st.sidebar.caption(f"Generated: {data['manifest'].get('generated_at')}")

    if page == "Forecasts":
        page_forecasts(data)
    elif page == "Diagnostics":
        page_diagnostics(data)
    else:
        page_backtest(data)


main()

"""Workhorse LightGBM quantile model + leak-safe purged walk-forward
evaluation.

``WorkhorseModel`` fits one ``LGBMRegressor(objective="quantile")`` per
requested quantile and exposes non-crossing predictions. ``run_walk_forward``
drives it through :func:`stockpred.validation.purged_walk_forward`, refitting
:class:`~stockpred.features.macro_pca.MacroFactorExtractor` on train-only
months each fold so no macro-factor information leaks from validation/test
months into the features.

Calibration split (per fold, for Task 7's conformal layer)
------------------------------------------------------------
Each fold's *train* months (already embargoed against its test block by
``purged_walk_forward``) are further split into:

- ``fit`` months: all train months except the most recent ``CAL_MONTHS``.
- ``cal`` months: the most recent ``CAL_MONTHS`` train months.

A first pair of models (quantile ``WorkhorseModel`` + a train-only-fit
``MacroFactorExtractor``) is fit on ``fit`` months only and used to predict
the ``cal`` months; those predictions are stored in ``WalkForwardResult.cal``
and are meant for downstream conformal calibration (Task 7) -- they are
genuinely out-of-sample for the models that produced them.

A second pair of models is then fit on the fold's *entire* train set
(``fit`` + ``cal`` months) -- maximizing training data -- and used to predict
the fold's test block; those predictions are stored in
``WalkForwardResult.oos``. This means ``cal`` and ``oos`` predictions come
from two different model fits, which is intentional: it keeps the
calibration set genuinely held-out while still using all available training
data for the headline out-of-sample test predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from stockpred.config import Config
from stockpred.features.macro_pca import MacroFactorExtractor
from stockpred.validation import decay_weights, purged_walk_forward

STOCK_FEATURE_COLS: list[str] = [
    "ret_1m",
    "mom_1m",
    "mom_3m",
    "mom_6m",
    "mom_12m",
    "mom_12_1",
    "vol_1m",
    "vol_3m",
    "vol_12m",
    "amihud",
    "dollar_vol",
    "beta_12m",
]

# Default n_pca_factors is 5 (see Config); the constant reflects that
# default. run_walk_forward itself derives the factor columns from
# cfg.n_pca_factors so it stays correct if that's overridden.
FEATURE_COLS: list[str] = STOCK_FEATURE_COLS + [f"F{i + 1}" for i in range(5)]

_DEFAULT_LGB_PARAMS: dict = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}

# Number of trailing train months held out per fold as the calibration set
# for Task 7's conformal layer.
CAL_MONTHS = 24
# Minimum number of fit-only months required after carving out CAL_MONTHS.
_MIN_FIT_MONTHS = 12

_SEED = 0


def _quantile_col(q: float) -> str:
    """Column name for quantile level ``q`` (e.g. 0.05 -> ``"q05"``).

    ``round`` rather than ``int``: ``0.29 * 100`` is ``28.999999999999996``
    in binary floating point, which ``int()`` truncates to 28. Every
    supported level is exact under ``round``.
    """
    return f"q{int(round(q * 100)):02d}"


def _macro_factor_cols(n_factors: int) -> list[str]:
    return [f"F{i + 1}" for i in range(n_factors)]


class WorkhorseModel:
    """One LightGBM quantile regressor per requested quantile.

    ``.fit`` trains an independent ``LGBMRegressor(objective="quantile",
    alpha=q)`` for each quantile in ``quantiles``. ``.predict`` returns a
    DataFrame with columns ``q05, q25, q50, q75, q95`` (named
    ``f"q{int(q*100):02d}"``), rows aligned to the input index, with
    non-crossing enforced by sorting each row's predicted values ascending
    across the (quantile-sorted) columns.
    """

    def __init__(self, quantiles: list[float], lgb_params: dict | None = None):
        self.quantiles = list(quantiles)
        merged = dict(_DEFAULT_LGB_PARAMS)
        if lgb_params:
            merged.update(lgb_params)
        self.lgb_params = merged
        self.models_: dict[float, LGBMRegressor] = {}

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> "WorkhorseModel":
        self.models_ = {}
        for q in self.quantiles:
            model = LGBMRegressor(
                objective="quantile",
                alpha=q,
                importance_type="gain",
                random_state=_SEED,
                deterministic=True,
                verbosity=-1,
                **self.lgb_params,
            )
            model.fit(X, y, sample_weight=sample_weight)
            self.models_[q] = model
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        quantiles_sorted = sorted(self.quantiles)
        cols_sorted = [_quantile_col(q) for q in quantiles_sorted]
        raw = np.column_stack([self.models_[q].predict(X) for q in quantiles_sorted])
        raw.sort(axis=1)  # enforce non-crossing
        return pd.DataFrame(raw, index=X.index, columns=cols_sorted)


@dataclass
class WalkForwardResult:
    oos: pd.DataFrame
    cal: pd.DataFrame
    metrics: dict
    feature_importance: pd.DataFrame


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def _predictions_frame(rows: pd.DataFrame, preds: pd.DataFrame, fold_id: int) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ticker": rows["ticker"].to_numpy(),
            "date": rows["date"].to_numpy(),
        }
    )
    for col in preds.columns:
        out[col] = preds[col].to_numpy()
    out["y_true"] = rows["fwd_ret_1m"].to_numpy()
    out["fold"] = fold_id
    return out


def _merge_factors(rows: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    return rows.merge(factors, left_on="date", right_index=True, how="left")


def run_walk_forward(
    panel: pd.DataFrame,
    macro_wide: pd.DataFrame,
    cfg: Config,
    lgb_params: dict | None = None,
) -> WalkForwardResult:
    """Leak-safe purged walk-forward evaluation of :class:`WorkhorseModel`.

    See the module docstring for the fit/cal/oos split semantics. Raises
    ``ValueError`` if any fold's train months are too few to carve out
    ``CAL_MONTHS`` calibration months plus a minimum fit window.
    """
    panel = panel.reset_index(drop=True)
    feature_cols = STOCK_FEATURE_COLS + _macro_factor_cols(cfg.n_pca_factors)

    oos_frames: list[pd.DataFrame] = []
    cal_frames: list[pd.DataFrame] = []
    fold_metrics: dict[int, dict] = {}
    importances: list[pd.Series] = []

    total_sse_q50 = 0.0
    total_sse_bench = 0.0

    folds = purged_walk_forward(
        panel["date"],
        n_folds=cfg.n_folds,
        test_window_months=cfg.test_window_months,
        embargo_months=cfg.embargo_months,
    )

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        train_dates = panel["date"].iloc[train_idx]
        unique_train_months = pd.DatetimeIndex(sorted(train_dates.unique()))

        if len(unique_train_months) < CAL_MONTHS + _MIN_FIT_MONTHS:
            raise ValueError(
                f"fold {fold_id}: only {len(unique_train_months)} train months "
                f"available, need at least {CAL_MONTHS + _MIN_FIT_MONTHS} "
                f"({CAL_MONTHS} calibration + {_MIN_FIT_MONTHS} minimum fit)"
            )

        cal_months = unique_train_months[-CAL_MONTHS:]
        fit_months = unique_train_months[:-CAL_MONTHS]

        train_dates_arr = train_dates.to_numpy()
        fit_idx = train_idx[np.isin(train_dates_arr, fit_months.to_numpy())]
        cal_idx = train_idx[np.isin(train_dates_arr, cal_months.to_numpy())]

        # --- Stage A: fit-only models -> calibration predictions ---
        extractor_fit = MacroFactorExtractor(cfg.n_pca_factors).fit(
            macro_wide.loc[macro_wide.index.isin(fit_months)]
        )
        factors_fit = extractor_fit.transform(macro_wide)

        fit_rows = _merge_factors(panel.iloc[fit_idx], factors_fit)
        X_fit = fit_rows[feature_cols]
        y_fit = fit_rows["fwd_ret_1m"]
        w_fit = decay_weights(fit_rows["date"], cfg.decay_half_life_years)

        model_fit = WorkhorseModel(cfg.quantiles, lgb_params=lgb_params)
        model_fit.fit(X_fit, y_fit, sample_weight=w_fit)

        cal_rows = _merge_factors(panel.iloc[cal_idx], factors_fit)
        cal_preds = model_fit.predict(cal_rows[feature_cols])
        cal_frames.append(_predictions_frame(cal_rows, cal_preds, fold_id))

        # --- Stage B: fit+cal models -> test (oos) predictions ---
        extractor_train = MacroFactorExtractor(cfg.n_pca_factors).fit(
            macro_wide.loc[macro_wide.index.isin(unique_train_months)]
        )
        factors_train = extractor_train.transform(macro_wide)

        train_rows = _merge_factors(panel.iloc[train_idx], factors_train)
        X_train = train_rows[feature_cols]
        y_train = train_rows["fwd_ret_1m"]
        w_train = decay_weights(train_rows["date"], cfg.decay_half_life_years)

        model_train = WorkhorseModel(cfg.quantiles, lgb_params=lgb_params)
        model_train.fit(X_train, y_train, sample_weight=w_train)

        test_rows = _merge_factors(panel.iloc[test_idx], factors_train)
        test_preds = model_train.predict(test_rows[feature_cols])
        oos_frames.append(_predictions_frame(test_rows, test_preds, fold_id))

        # --- Metrics: r2_oos_median vs historical-mean benchmark ---
        benchmark_mean = float(y_train.mean())
        y_test = test_rows["fwd_ret_1m"].to_numpy()
        q50_pred = test_preds["q50"].to_numpy()
        sse_q50 = float(np.sum((y_test - q50_pred) ** 2))
        sse_bench = float(np.sum((y_test - benchmark_mean) ** 2))
        fold_r2 = 1.0 - sse_q50 / sse_bench if sse_bench > 0 else float("nan")
        total_sse_q50 += sse_q50
        total_sse_bench += sse_bench

        fold_pinball = {
            _quantile_col(q): _pinball_loss(y_test, test_preds[_quantile_col(q)].to_numpy(), q)
            for q in cfg.quantiles
        }
        fold_coverage = float(
            np.mean((test_preds["q05"].to_numpy() <= y_test) & (y_test <= test_preds["q95"].to_numpy()))
        )
        fold_metrics[fold_id] = {
            "r2_oos_median": fold_r2,
            "pinball_loss": fold_pinball,
            "coverage_90": fold_coverage,
            "n_train_months": len(unique_train_months),
            "n_test_rows": len(test_rows),
        }

        q50_model = model_train.models_[0.5]
        importances.append(pd.Series(q50_model.feature_importances_, index=feature_cols))

    if not oos_frames:
        raise ValueError(
            "run_walk_forward: no folds produced -- panel/macro_wide/cfg "
            "don't cover enough months for the configured n_folds/"
            "test_window_months/embargo_months"
        )

    oos = pd.concat(oos_frames, ignore_index=True)
    cal = pd.concat(cal_frames, ignore_index=True)

    pooled_pinball = {
        _quantile_col(q): _pinball_loss(
            oos["y_true"].to_numpy(), oos[_quantile_col(q)].to_numpy(), q
        )
        for q in cfg.quantiles
    }
    pooled_coverage = float(
        np.mean((oos["q05"].to_numpy() <= oos["y_true"].to_numpy())
                & (oos["y_true"].to_numpy() <= oos["q95"].to_numpy()))
    )
    pooled_r2 = 1.0 - total_sse_q50 / total_sse_bench if total_sse_bench > 0 else float("nan")

    metrics = {
        "pooled": {
            "r2_oos_median": pooled_r2,
            "pinball_loss": pooled_pinball,
            "coverage_90": pooled_coverage,
        },
        "folds": fold_metrics,
    }

    importance_df = (
        pd.concat(importances, axis=1)
        .mean(axis=1)
        .rename("importance")
        .rename_axis("feature")
        .reset_index()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return WalkForwardResult(
        oos=oos, cal=cal, metrics=metrics, feature_importance=importance_df
    )

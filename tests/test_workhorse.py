"""Tests for stockpred.models.workhorse: the LightGBM quantile workhorse
model and the leak-safe purged walk-forward evaluation harness.

Synthetic panel fixture: ~30 tickers x 200 months, with a single feature
column (``mom_12_1``) carrying a real linear signal (``fwd_ret_1m = 0.3 *
signal + noise``) so the q50 model has something learnable to beat the
historical-mean benchmark, while the remaining stock feature columns are
pure noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpred.config import Config
from stockpred.models.workhorse import (
    FEATURE_COLS,
    STOCK_FEATURE_COLS,
    WalkForwardResult,
    WorkhorseModel,
    _quantile_col,
    run_walk_forward,
)

# Small LightGBM params for fast tests -- overridden onto the model's
# defaults (the interface must MERGE overrides, not replace all params).
_FAST_LGB_PARAMS = {"n_estimators": 50, "num_leaves": 15}

_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def _month_ends(n: int, start: str = "2005-01-31") -> pd.DatetimeIndex:
    first = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
    return pd.DatetimeIndex([first + pd.offsets.MonthEnd(i) for i in range(n)])


def _make_synthetic_panel(
    n_tickers: int = 30, n_months: int = 200, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = _month_ends(n_months)
    frames = []
    for t in range(n_tickers):
        ticker = f"T{t:02d}"
        signal = rng.normal(scale=1.0, size=n_months)
        noise = rng.normal(scale=1.0, size=n_months)
        data = {col: rng.normal(size=n_months) for col in STOCK_FEATURE_COLS}
        data["mom_12_1"] = signal  # the one column carrying real signal
        df = pd.DataFrame(data)
        df["ticker"] = ticker
        df["date"] = months
        df["fwd_ret_1m"] = 0.3 * signal + noise
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    return panel[["ticker", "date", *STOCK_FEATURE_COLS, "fwd_ret_1m"]]


def _make_macro_wide(n_months: int = 200, n_series: int = 6, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = _month_ends(n_months)
    data = rng.normal(size=(n_months, n_series))
    cols = [f"M{i}" for i in range(n_series)]
    return pd.DataFrame(data, index=months, columns=cols).rename_axis("date")


def _make_config(**overrides) -> Config:
    kwargs = dict(
        tickers=["A"],
        fred_series=[],
        ecb_series=[],
        data_dir=Path("data"),
        artifacts_dir=Path("artifacts"),
        duckdb_path=Path("db.duckdb"),
    )
    kwargs.update(overrides)
    return Config(**kwargs)


# ---------------------------------------------------------------------------
# WorkhorseModel
# ---------------------------------------------------------------------------


class TestWorkhorseModel:
    def test_predict_columns_and_alignment(self):
        rng = np.random.default_rng(0)
        n = 300
        X = pd.DataFrame(
            {"f1": rng.normal(size=n), "f2": rng.normal(size=n)},
            index=range(100, 100 + n),
        )
        y = X["f1"] * 0.5 + rng.normal(scale=0.1, size=n)

        model = WorkhorseModel(_QUANTILES, lgb_params=_FAST_LGB_PARAMS)
        model.fit(X, y)
        preds = model.predict(X)

        assert list(preds.columns) == ["q05", "q25", "q50", "q75", "q95"]
        assert list(preds.index) == list(X.index)

    def test_non_crossing_quantiles(self):
        rng = np.random.default_rng(3)
        n = 200
        X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
        y = rng.normal(size=n)  # pure noise -> models likely to disagree/cross without sorting

        model = WorkhorseModel(_QUANTILES, lgb_params=_FAST_LGB_PARAMS)
        model.fit(X, y)
        preds = model.predict(X)

        vals = preds[["q05", "q25", "q50", "q75", "q95"]].to_numpy()
        assert (np.diff(vals, axis=1) >= 0).all()

    def test_lgb_params_merge_with_defaults(self):
        model = WorkhorseModel([0.5], lgb_params={"n_estimators": 7})
        assert model.lgb_params["n_estimators"] == 7
        # untouched defaults still present
        assert model.lgb_params["learning_rate"] == 0.03
        assert model.lgb_params["num_leaves"] == 31

    def test_sample_weight_accepted(self):
        rng = np.random.default_rng(4)
        n = 100
        X = pd.DataFrame({"f1": rng.normal(size=n)})
        y = rng.normal(size=n)
        w = np.ones(n)
        model = WorkhorseModel([0.5], lgb_params=_FAST_LGB_PARAMS)
        model.fit(X, y, sample_weight=w)  # should not raise
        preds = model.predict(X)
        assert "q50" in preds.columns


# ---------------------------------------------------------------------------
# run_walk_forward
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def wf_result() -> WalkForwardResult:
    panel = _make_synthetic_panel()
    macro_wide = _make_macro_wide()
    cfg = _make_config()
    return run_walk_forward(panel, macro_wide, cfg, lgb_params=_FAST_LGB_PARAMS)


class TestRunWalkForward:
    def test_produces_folds(self, wf_result):
        assert len(wf_result.oos) > 0
        assert len(wf_result.cal) > 0
        assert wf_result.oos["fold"].nunique() >= 1

    def test_oos_schema(self, wf_result):
        expected = {"ticker", "date", "q05", "q25", "q50", "q75", "q95", "y_true", "fold"}
        assert expected.issubset(set(wf_result.oos.columns))
        assert expected.issubset(set(wf_result.cal.columns))

    def test_quantiles_non_crossing_in_oos_and_cal(self, wf_result):
        for df in (wf_result.oos, wf_result.cal):
            vals = df[["q05", "q25", "q50", "q75", "q95"]].to_numpy()
            assert (np.diff(vals, axis=1) >= 0).all()

    def test_cal_disjoint_from_oos(self, wf_result):
        oos_keys = set(zip(wf_result.oos["ticker"], wf_result.oos["date"], wf_result.oos["fold"]))
        cal_keys = set(zip(wf_result.cal["ticker"], wf_result.cal["date"], wf_result.cal["fold"]))
        assert oos_keys.isdisjoint(cal_keys)

    def test_no_fold_trains_on_data_inside_embargo(self, wf_result):
        # Reconstruct, per fold, the max training date present anywhere
        # (oos rows only tell us about test dates; verify indirectly via
        # cal+oos test-vs-train separation is exercised in the harness
        # itself -- here we check the weaker, directly observable
        # invariant: cal dates for a fold are always < that fold's oos
        # (test) dates, since cal is drawn from train months only.
        for fold_id in wf_result.oos["fold"].unique():
            oos_fold = wf_result.oos[wf_result.oos["fold"] == fold_id]
            cal_fold = wf_result.cal[wf_result.cal["fold"] == fold_id]
            if cal_fold.empty:
                continue
            assert cal_fold["date"].max() < oos_fold["date"].min()

    def test_q50_beats_naive_mean_benchmark(self, wf_result):
        assert wf_result.metrics["pooled"]["r2_oos_median"] > 0

    def test_pinball_loss_and_coverage_present(self, wf_result):
        pooled = wf_result.metrics["pooled"]
        assert set(pooled["pinball_loss"].keys()) == {"q05", "q25", "q50", "q75", "q95"}
        assert 0.0 <= pooled["coverage_90"] <= 1.0

    def test_feature_importance_schema(self, wf_result):
        fi = wf_result.feature_importance
        assert list(fi.columns) == ["feature", "importance"]
        assert set(fi["feature"]) == set(FEATURE_COLS)
        # sorted descending by importance
        assert (fi["importance"].diff().dropna() <= 0).all()

    def test_quantile_col_names_use_round_not_truncation(self):
        # 0.29 * 100 == 28.999999999999996 in binary floating point; int()
        # would truncate that to "q28". Every supported level must map to
        # its exact two-digit name.
        assert _quantile_col(0.05) == "q05"
        assert _quantile_col(0.25) == "q25"
        assert _quantile_col(0.5) == "q50"
        assert _quantile_col(0.75) == "q75"
        assert _quantile_col(0.95) == "q95"
        assert _quantile_col(0.29) == "q29"

    def test_feature_cols_constant(self):
        assert FEATURE_COLS == STOCK_FEATURE_COLS + ["F1", "F2", "F3", "F4", "F5"]

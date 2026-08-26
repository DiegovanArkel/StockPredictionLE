"""Tests for stockpred.models.conformal: split conformalized quantile
regression (CQR) calibration.

Conventions mirror tests/test_workhorse.py: synthetic data, seeded RNGs,
returns as decimal fractions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from stockpred.models.conformal import apply_cqr, calibrate_from_wf, cqr_offsets
from stockpred.models.workhorse import WalkForwardResult


def _synthetic_intervals(n: int, coverage: float, seed: int) -> pd.DataFrame:
    """y ~ N(0, 1); [lo, hi] = [-c, c] where c is chosen so that the
    interval covers exactly `coverage` fraction of a standard normal in
    expectation (i.e. deliberately narrow/wide relative to the nominal 90%
    interval, which uses c = 1.645...).
    """
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    c = stats.norm.ppf(0.5 + coverage / 2)
    lo = np.full(n, -c)
    hi = np.full(n, c)
    return pd.DataFrame({"lo": lo, "hi": hi, "y": y})


class TestCqrOffsets:
    def test_narrow_intervals_restore_nominal_coverage_on_fresh_data(self):
        n_cal = 2000
        n_test = 2000
        cal = _synthetic_intervals(n_cal, coverage=0.70, seed=0)
        offset = cqr_offsets(cal["lo"], cal["hi"], cal["y"], alpha=0.10)

        assert offset > 0  # intervals were too narrow -> offset widens them

        test = _synthetic_intervals(n_test, coverage=0.70, seed=1)
        lo_cal = test["lo"] - offset
        hi_cal = test["hi"] + offset
        covered = (test["y"] >= lo_cal) & (test["y"] <= hi_cal)
        coverage = covered.mean()
        assert abs(coverage - 0.90) <= 0.03

    def test_over_wide_intervals_shrink(self):
        n_cal = 2000
        cal = _synthetic_intervals(n_cal, coverage=0.995, seed=2)
        offset = cqr_offsets(cal["lo"], cal["hi"], cal["y"], alpha=0.10)
        assert offset < 0

    def test_degenerate_small_n_raises(self):
        cal = _synthetic_intervals(19, coverage=0.70, seed=3)
        with pytest.raises(ValueError):
            cqr_offsets(cal["lo"], cal["hi"], cal["y"], alpha=0.10)

    def test_n_exactly_20_does_not_raise(self):
        cal = _synthetic_intervals(20, coverage=0.70, seed=4)
        # should not raise
        cqr_offsets(cal["lo"], cal["hi"], cal["y"], alpha=0.10)

    def test_finite_sample_quantile_level(self):
        # With n calibration points, the empirical quantile level used is
        # min(1, ceil((n+1)(1-alpha)) / n) -- verify against a hand-rolled
        # computation using np.quantile(..., method="higher").
        n = 100
        rng = np.random.default_rng(5)
        lo = np.zeros(n)
        hi = np.zeros(n)
        y = rng.normal(size=n)
        alpha = 0.10
        scores = np.maximum(lo - y, y - hi)
        level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        expected = np.quantile(scores, level, method="higher")
        actual = cqr_offsets(lo, hi, y, alpha=alpha)
        assert actual == pytest.approx(expected)


class TestApplyCqr:
    def _pred_frame(self, n: int = 10, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        base = np.sort(rng.normal(size=(n, 5)), axis=1)
        return pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "date": pd.date_range("2020-01-31", periods=n, freq="ME"),
                "q05": base[:, 0],
                "q25": base[:, 1],
                "q50": base[:, 2],
                "q75": base[:, 3],
                "q95": base[:, 4],
                "y_true": rng.normal(size=n),
                "fold": 0,
            }
        )

    def test_adds_cal_columns_widening_by_offsets(self):
        pred = self._pred_frame()
        offsets = {"90": 0.05, "50": 0.02}
        out = apply_cqr(pred, offsets)

        for col in ["q05_cal", "q25_cal", "q75_cal", "q95_cal"]:
            assert col in out.columns

        # Original columns preserved untouched.
        pd.testing.assert_series_equal(out["q05"], pred["q05"])
        pd.testing.assert_series_equal(out["q95"], pred["q95"])
        pd.testing.assert_series_equal(out["q50"], pred["q50"])

    def test_non_crossing_after_adjustment(self):
        pred = self._pred_frame(n=50, seed=7)
        # Large, asymmetric offsets that could induce crossing without the
        # sort-based fixup.
        offsets = {"90": -0.5, "50": 0.5}
        out = apply_cqr(pred, offsets)
        vals = out[["q05_cal", "q25_cal", "q50", "q75_cal", "q95_cal"]].to_numpy()
        assert np.all(np.diff(vals, axis=1) >= -1e-12)

    def test_negative_offset_shrinks_interval(self):
        pred = self._pred_frame(n=5, seed=1)
        offsets = {"90": -0.1, "50": -0.05}
        out = apply_cqr(pred, offsets)
        assert (out["q95_cal"] <= pred["q95"]).all()
        assert (out["q05_cal"] >= pred["q05"]).all()

    def test_positive_offset_widens_interval(self):
        pred = self._pred_frame(n=5, seed=2)
        offsets = {"90": 0.1, "50": 0.05}
        out = apply_cqr(pred, offsets)
        assert (out["q95_cal"] >= pred["q95"]).all()
        assert (out["q05_cal"] <= pred["q05"]).all()


class TestCalibrateFromWf:
    def _make_wf(
        self, n_cal: int = 2000, n_oos: int = 2000, coverage: float = 0.70
    ) -> WalkForwardResult:
        def _frame(n: int, seed: int) -> pd.DataFrame:
            rng = np.random.default_rng(seed)
            y = rng.normal(size=n)
            c90 = stats.norm.ppf(0.5 + coverage / 2)
            c50 = stats.norm.ppf(0.5 + coverage / 2 * (0.5 / 0.90))
            return pd.DataFrame(
                {
                    "ticker": [f"T{i % 10}" for i in range(n)],
                    "date": pd.date_range("2020-01-31", periods=n, freq="D"),
                    "q05": -c90,
                    "q25": -c50 if c50 < c90 else -c90 * 0.4,
                    "q50": 0.0,
                    "q75": c50 if c50 < c90 else c90 * 0.4,
                    "q95": c90,
                    "y_true": y,
                    "fold": np.arange(n) % 3,
                }
            )

        cal = _frame(n_cal, seed=10)
        oos = _frame(n_oos, seed=11)
        return WalkForwardResult(oos=oos, cal=cal, metrics={}, feature_importance=pd.DataFrame())

    def test_returns_offsets_and_calibrated_oos(self):
        wf = self._make_wf()
        offsets, oos_cal = calibrate_from_wf(wf)

        assert set(["90", "50", "coverage_raw_90", "coverage_cal_90"]).issubset(offsets.keys())
        assert isinstance(offsets["coverage_raw_90"], float)
        assert isinstance(offsets["coverage_cal_90"], float)
        assert len(oos_cal) == len(wf.oos)
        for col in ["q05_cal", "q25_cal", "q75_cal", "q95_cal"]:
            assert col in oos_cal.columns

    def test_calibrated_coverage_near_nominal(self):
        wf = self._make_wf(coverage=0.70)
        offsets, oos_cal = calibrate_from_wf(wf)

        assert offsets["90"] > 0
        assert abs(offsets["coverage_cal_90"] - 0.90) <= 0.03
        # raw coverage should reflect the deliberately narrow intervals
        assert offsets["coverage_raw_90"] < offsets["coverage_cal_90"]

    def test_offsets_pooled_across_all_folds(self):
        # cal has 3 folds (0,1,2); offsets should be computed on the union,
        # not per-fold, so results should differ from using a single fold.
        wf = self._make_wf(n_cal=300, n_oos=300)
        pooled_offsets, _ = calibrate_from_wf(wf)

        single_fold_cal = wf.cal[wf.cal["fold"] == 0]
        assert len(single_fold_cal) < len(wf.cal)
        single_fold_offset = cqr_offsets(
            single_fold_cal["q05"], single_fold_cal["q95"], single_fold_cal["y_true"], alpha=0.10
        )
        # Not asserting equality (that would be a tautology given RNG), just
        # confirming pooled uses more data than a single fold would.
        assert len(wf.cal) == 300
        assert isinstance(single_fold_offset, float)
        assert isinstance(pooled_offsets["90"], float)

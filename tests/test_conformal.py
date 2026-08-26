"""Tests for stockpred.models.conformal: split conformalized quantile
regression (CQR) calibration.

Conventions mirror tests/test_workhorse.py: synthetic data, seeded RNGs,
returns as decimal fractions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from stockpred.config import Config
from stockpred.models.conformal import apply_cqr, calibrate_from_wf, cqr_offsets
from stockpred.models.workhorse import (
    STOCK_FEATURE_COLS,
    WalkForwardResult,
    run_walk_forward,
)


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

    def _make_wf_fold_dependent(self, n_per_fold: int = 400) -> WalkForwardResult:
        """Three folds whose calibration residual *scale* differs sharply by
        fold (fold f's y_true has std 1 + f), so each fold's own CQR offset
        must differ from the others' and from the pooled offset.
        """
        rng = np.random.default_rng(21)

        def _frame(seed_offset: int) -> pd.DataFrame:
            rows = []
            for fold in range(3):
                r = np.random.default_rng(100 * fold + seed_offset)
                rows.append(
                    pd.DataFrame(
                        {
                            "ticker": [f"T{i % 5}" for i in range(n_per_fold)],
                            "date": pd.date_range(
                                f"20{20 + fold}-01-31", periods=n_per_fold, freq="D"
                            ),
                            "q05": -1.0,
                            "q25": -0.4,
                            "q50": 0.0,
                            "q75": 0.4,
                            "q95": 1.0,
                            "y_true": r.normal(scale=1.0 + fold, size=n_per_fold),
                            "fold": fold,
                        }
                    )
                )
            return pd.concat(rows, ignore_index=True)

        _ = rng  # keep the seeded-RNG convention explicit
        return WalkForwardResult(
            oos=_frame(1), cal=_frame(2), metrics={}, feature_importance=pd.DataFrame()
        )

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

    def test_per_fold_offsets_differ_across_folds(self):
        # Each fold's calibration residuals have a different scale, so each
        # fold's own CQR offset must differ materially from the others'.
        wf = self._make_wf_fold_dependent()
        offsets, _ = calibrate_from_wf(wf)

        per_fold = offsets["per_fold"]
        assert set(per_fold) == {"0", "1", "2"}

        o90 = [per_fold[str(f)]["90"] for f in range(3)]
        # Wider residuals -> larger widening offset, strictly monotone here.
        assert o90[0] < o90[1] < o90[2]
        # And each is meaningfully apart, not floating-point noise.
        assert o90[1] - o90[0] > 0.5
        assert o90[2] - o90[1] > 0.5

    def test_each_folds_oos_calibrated_by_its_own_offset(self):
        # The real assertion the old pooled test was missing: fold k's
        # calibrated bounds must equal fold k's RAW bounds shifted by fold
        # k's OWN offset -- not by the pooled offset.
        wf = self._make_wf_fold_dependent()
        offsets, oos_cal = calibrate_from_wf(wf)

        for fold in range(3):
            sub = oos_cal[oos_cal["fold"] == fold]
            own = offsets["per_fold"][str(fold)]
            assert (sub["q95_cal"] - sub["q95"]).abs().max() == pytest.approx(
                abs(own["90"]), abs=1e-12
            )
            np.testing.assert_allclose(
                sub["q95_cal"].to_numpy(), sub["q95"].to_numpy() + own["90"], atol=1e-12
            )
            np.testing.assert_allclose(
                sub["q05_cal"].to_numpy(), sub["q05"].to_numpy() - own["90"], atol=1e-12
            )
            # ...and NOT by the pooled offset (which really is different).
            assert abs(own["90"] - offsets["90"]) > 1e-6

    def test_pooled_offsets_reported_but_not_used_for_calibration(self):
        wf = self._make_wf_fold_dependent()
        offsets, oos_cal = calibrate_from_wf(wf)

        pooled_90 = cqr_offsets(
            wf.cal["q05"], wf.cal["q95"], wf.cal["y_true"], alpha=0.10
        )
        assert offsets["90"] == pytest.approx(pooled_90)

        # If the pooled offset had been applied to every fold, every fold's
        # widening would be identical; per-fold calibration means it isn't.
        widenings = {
            fold: float((oos_cal.loc[oos_cal["fold"] == fold, "q95_cal"]
                         - oos_cal.loc[oos_cal["fold"] == fold, "q95"]).iloc[0])
            for fold in range(3)
        }
        assert len(set(round(v, 9) for v in widenings.values())) == 3

    def test_row_order_and_length_preserved(self):
        wf = self._make_wf_fold_dependent()
        _, oos_cal = calibrate_from_wf(wf)
        assert len(oos_cal) == len(wf.oos)
        pd.testing.assert_series_equal(oos_cal["fold"], wf.oos["fold"])
        pd.testing.assert_series_equal(oos_cal["y_true"], wf.oos["y_true"])


# ---------------------------------------------------------------------------
# Leakage regression: the project's core promise
# ---------------------------------------------------------------------------


def _month_ends(n: int, start: str = "2005-01-31") -> pd.DatetimeIndex:
    first = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
    return pd.DatetimeIndex([first + pd.offsets.MonthEnd(i) for i in range(n)])


def _small_panel(n_tickers: int = 6, n_months: int = 170, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = _month_ends(n_months)
    frames = []
    for t in range(n_tickers):
        signal = rng.normal(size=n_months)
        data = {col: rng.normal(size=n_months) for col in STOCK_FEATURE_COLS}
        data["mom_12_1"] = signal
        df = pd.DataFrame(data)
        df["ticker"] = f"T{t:02d}"
        df["date"] = months
        df["fwd_ret_1m"] = 0.3 * signal + rng.normal(size=n_months)
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    return panel[["ticker", "date", *STOCK_FEATURE_COLS, "fwd_ret_1m"]]


def _small_macro_wide(n_months: int = 170, n_series: int = 6, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = _month_ends(n_months)
    return pd.DataFrame(
        rng.normal(size=(n_months, n_series)),
        index=months,
        columns=[f"M{i}" for i in range(n_series)],
    ).rename_axis("date")


def _cfg(**overrides) -> Config:
    kwargs = dict(
        tickers=["A"],
        fred_series=[],
        ecb_series=[],
        data_dir=Path("data"),
        artifacts_dir=Path("artifacts"),
        duckdb_path=Path("db.duckdb"),
        n_folds=3,
    )
    kwargs.update(overrides)
    return Config(**kwargs)


@pytest.fixture(scope="module")
def _wf_real_harness() -> WalkForwardResult:
    return run_walk_forward(
        _small_panel(),
        _small_macro_wide(),
        _cfg(),
        lgb_params={"n_estimators": 25, "num_leaves": 7},
    )


class TestNoTemporalLeakageInCalibration:
    """The core promise of this project: nothing that calibrates a fold's
    reported intervals may be dated at or after that fold's test months.

    Model disjointness (``cal`` rows never being used to *fit* the model
    that produced the ``oos`` predictions) is NOT sufficient -- the CQR
    offset is itself a statistic computed from data, and it must be a
    statistic of the past. This test checks the temporal property directly.
    """

    def test_every_folds_calibration_rows_strictly_precede_its_oos_rows(
        self, _wf_real_harness
    ):
        wf = _wf_real_harness
        offsets, _ = calibrate_from_wf(wf)

        fold_ids = sorted(wf.oos["fold"].unique())
        assert len(fold_ids) >= 2, "need multiple folds for this to mean anything"
        assert set(offsets["per_fold"]) == {str(f) for f in fold_ids}

        for fold_id in fold_ids:
            cal_rows = wf.cal[wf.cal["fold"] == fold_id]
            oos_rows = wf.oos[wf.oos["fold"] == fold_id]
            assert not cal_rows.empty and not oos_rows.empty

            last_cal_date = pd.Timestamp(cal_rows["date"].max())
            first_oos_date = pd.Timestamp(oos_rows["date"].min())
            assert last_cal_date < first_oos_date, (
                f"fold {fold_id}: calibration data up to {last_cal_date} was used "
                f"to calibrate test months starting {first_oos_date} -- look-ahead leak"
            )

    def test_pooled_calibration_would_have_leaked(self, _wf_real_harness):
        """Guard the guard: confirm the pooled alternative really does
        violate the property above (otherwise the test could pass
        vacuously on a fixture where pooling happens to be harmless).
        """
        wf = _wf_real_harness
        pooled_last_cal = pd.Timestamp(wf.cal["date"].max())
        earliest_oos = pd.Timestamp(wf.oos["date"].min())
        assert pooled_last_cal > earliest_oos

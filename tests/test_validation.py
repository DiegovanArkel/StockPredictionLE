"""Tests for stockpred.validation: purged walk-forward CV, decay weights,
and the deflated Sharpe ratio.

No look-ahead constraint under test: embargo means train months must satisfy
``train_month <= test_start_month - (embargo_months + 1) months`` (the extra
month buys back the 1-month forecast horizon: the target at train month t is
realized during t+1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.validation import decay_weights, deflated_sharpe_ratio, purged_walk_forward


def _month_ends(n_months: int, start: str = "2000-01-31") -> list[pd.Timestamp]:
    first = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
    return [first + pd.offsets.MonthEnd(i) for i in range(n_months)]


def _make_panel_dates(n_months: int, n_tickers: int = 3, start: str = "2000-01-31") -> pd.Series:
    """A monthly `date` column as it appears in the real panel: every
    month-end date repeated once per ticker."""
    months = _month_ends(n_months, start=start)
    dates = [m for m in months for _ in range(n_tickers)]
    return pd.Series(dates)


class TestPurgedWalkForward:
    def test_embargo_respected(self):
        dates = _make_panel_dates(60, n_tickers=2)
        embargo_months = 2
        folds = list(
            purged_walk_forward(
                dates,
                n_folds=3,
                test_window_months=6,
                embargo_months=embargo_months,
                min_train_months=6,
            )
        )
        assert len(folds) == 3
        for train_idx, test_idx in folds:
            train_dates = dates.iloc[train_idx]
            test_dates = dates.iloc[test_idx]
            test_start = test_dates.min()
            # Independently derive the true month-end cutoff via Period
            # arithmetic, decoupled from whatever offset type the
            # implementation itself uses (pd.DateOffset(months=N) does NOT
            # preserve month-end alignment, e.g. 2020-09-30 - DateOffset
            # (months=1) == 2020-08-30, not 2020-08-31 -- so this must not
            # reuse that expression to avoid masking the same bug).
            cutoff_period = test_start.to_period("M") - (embargo_months + 1)
            cutoff = cutoff_period.to_timestamp(how="end").normalize()
            assert (train_dates <= cutoff).all()

    def test_embargo_cutoff_preserves_month_end_alignment(self):
        # Regression for a real bug: pd.DateOffset(months=N) subtracted from
        # a month-end Timestamp does not snap back to month-end when the
        # target month has a different length, silently dropping a whole
        # valid training month. Build a panel whose test block starts
        # exactly on 2020-09-30 (a 30-day month) with embargo_months=0, so
        # the true cutoff (embargo(0) + horizon(1) = 1 month back) must be
        # 2020-08-31 -- the true last day of August, not "day 30".
        dates = _make_panel_dates(33, n_tickers=2, start="2018-01-31")
        assert dates.max() == pd.Timestamp("2020-09-30")  # sanity on the fixture itself

        folds = list(
            purged_walk_forward(
                dates, n_folds=1, test_window_months=1, embargo_months=0, min_train_months=1
            )
        )
        assert len(folds) == 1
        train_idx, test_idx = folds[0]
        assert dates.iloc[test_idx].min() == pd.Timestamp("2020-09-30")
        last_train_month = dates.iloc[train_idx].max()
        assert last_train_month == pd.Timestamp("2020-08-31")

    def test_blocks_non_overlapping_consecutive_chronological(self):
        dates = _make_panel_dates(60, n_tickers=2)
        folds = list(
            purged_walk_forward(
                dates, n_folds=4, test_window_months=6, embargo_months=1, min_train_months=6
            )
        )
        assert len(folds) == 4
        test_month_blocks = []
        for _, test_idx in folds:
            months = sorted(dates.iloc[test_idx].unique())
            test_month_blocks.append(months)

        # each block has exactly test_window_months distinct months
        for block in test_month_blocks:
            assert len(block) == 6

        # chronological order across yielded folds
        for i in range(len(test_month_blocks) - 1):
            assert test_month_blocks[i][-1] < test_month_blocks[i + 1][0]

        # non-overlapping
        flat = [m for block in test_month_blocks for m in block]
        assert len(flat) == len(set(flat))

        # consecutive (no gap between one block's last month and the next's first)
        for i in range(len(test_month_blocks) - 1):
            end_of_block = test_month_blocks[i][-1]
            start_of_next = test_month_blocks[i + 1][0]
            assert start_of_next == end_of_block + pd.offsets.MonthEnd(1)

    def test_skips_folds_with_insufficient_train_history(self):
        dates = _make_panel_dates(20, n_tickers=2)
        folds = list(
            purged_walk_forward(
                dates,
                n_folds=5,
                test_window_months=3,
                embargo_months=1,
                min_train_months=100,
            )
        )
        assert folds == []

    def test_partial_folds_yielded_when_some_have_enough_train(self):
        # 60 months; min_train_months chosen so only the later folds qualify.
        dates = _make_panel_dates(60, n_tickers=2)
        folds = list(
            purged_walk_forward(
                dates,
                n_folds=4,
                test_window_months=6,
                embargo_months=1,
                min_train_months=40,
            )
        )
        assert 0 < len(folds) < 4
        for train_idx, _ in folds:
            assert len(np.unique(dates.iloc[train_idx])) >= 40

    def test_indices_cover_all_tickers_of_each_month(self):
        dates = _make_panel_dates(60, n_tickers=5)
        folds = list(
            purged_walk_forward(
                dates, n_folds=2, test_window_months=3, embargo_months=1, min_train_months=12
            )
        )
        assert len(folds) == 2
        for train_idx, test_idx in folds:
            for label, idx in (("train", train_idx), ("test", test_idx)):
                months_in_split = dates.iloc[idx].unique()
                for m in months_in_split:
                    expected = set(np.nonzero((dates == m).to_numpy())[0])
                    got = set(idx[dates.iloc[idx].to_numpy() == m])
                    assert got == expected, f"{label} split missing rows for month {m}"

    def test_train_and_test_disjoint(self):
        dates = _make_panel_dates(60, n_tickers=3)
        folds = list(
            purged_walk_forward(
                dates, n_folds=3, test_window_months=4, embargo_months=1, min_train_months=12
            )
        )
        assert len(folds) == 3
        for train_idx, test_idx in folds:
            assert set(train_idx).isdisjoint(set(test_idx))

    def test_indices_are_positional_not_label_based(self):
        dates = _make_panel_dates(60, n_tickers=2)
        dates.index = np.arange(1000, 1000 + len(dates))  # non-default label index
        folds = list(
            purged_walk_forward(
                dates, n_folds=2, test_window_months=3, embargo_months=1, min_train_months=12
            )
        )
        assert len(folds) == 2
        n = len(dates)
        for train_idx, test_idx in folds:
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert train_idx.min() >= 0
            assert test_idx.max() < n
            assert train_idx.max() < n

    def test_returns_generator(self):
        import types

        dates = _make_panel_dates(60, n_tickers=2)
        result = purged_walk_forward(
            dates, n_folds=2, test_window_months=3, embargo_months=1, min_train_months=12
        )
        assert isinstance(result, types.GeneratorType)


class TestDecayWeights:
    def test_mean_is_one(self):
        dates = pd.Series(_month_ends(48))
        w = decay_weights(dates, half_life_years=5.0)
        assert isinstance(w, np.ndarray)
        assert w.mean() == pytest.approx(1.0)

    def test_halves_at_half_life(self):
        dates = pd.Series(_month_ends(180))  # 15 years of month-ends
        half_life_years = 5.0
        w = decay_weights(dates, half_life_years=half_life_years)
        max_date = dates.max()
        target = max_date - pd.DateOffset(years=int(half_life_years))
        idx_max = int((dates == max_date).to_numpy().argmax())
        idx_target = int((dates - target).abs().to_numpy().argmin())
        ratio = w[idx_target] / w[idx_max]
        assert ratio == pytest.approx(0.5, rel=0.02)

    def test_monotonic_recency_preference(self):
        dates = pd.Series(_month_ends(24))
        w = decay_weights(dates, half_life_years=3.0)
        assert np.all(np.diff(w) >= 0)  # later dates => weight non-decreasing
        assert w[-1] > w[0]

    def test_most_recent_date_has_zero_age_full_weight_before_norm(self):
        # weight of the max date, relative to itself, should be the array max
        dates = pd.Series(_month_ends(36))
        w = decay_weights(dates, half_life_years=4.0)
        assert w.argmax() == len(dates) - 1


class TestDeflatedSharpeRatio:
    def test_n_trials_below_one_raises(self):
        with pytest.raises(ValueError):
            deflated_sharpe_ratio(np.array([0.01, 0.02, -0.01]), n_trials=0)

    def test_short_series_is_uninformative(self):
        result = deflated_sharpe_ratio(np.array([0.01, 0.02]), n_trials=5)
        assert result == {"sharpe": 0.0, "psr": 0.5, "dsr": 0.5}

    def test_zero_variance_is_uninformative(self):
        result = deflated_sharpe_ratio(np.zeros(10), n_trials=5)
        assert result == {"sharpe": 0.0, "psr": 0.5, "dsr": 0.5}

    def test_pure_noise_has_low_psr(self):
        # Genuine seeded zero-mean noise, no artificial recentering. This
        # seed happens to draw a slightly negative sample mean, giving a
        # clearly low (not just sub-0.5) PSR; the threshold is kept loose
        # (0.7) so the test isn't coupled to this exact seed's draw.
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.05, size=120)
        result = deflated_sharpe_ratio(returns, n_trials=8, periods_per_year=12)
        assert result["psr"] < 0.7

    def test_strong_signal_has_high_psr(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(0.02, 0.02, size=60)
        result = deflated_sharpe_ratio(returns, n_trials=8, periods_per_year=12)
        assert result["psr"] > 0.9
        assert result["dsr"] > 0.9

    def test_more_trials_lowers_or_holds_dsr(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(0.02, 0.02, size=60)
        few = deflated_sharpe_ratio(returns, n_trials=2, periods_per_year=12)
        many = deflated_sharpe_ratio(returns, n_trials=50, periods_per_year=12)
        assert many["dsr"] <= few["dsr"]
        assert many["psr"] == few["psr"] == few["psr"]  # psr independent of n_trials

    def test_dsr_le_psr(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0.015, 0.03, size=80)
        result = deflated_sharpe_ratio(returns, n_trials=10, periods_per_year=12)
        assert result["dsr"] <= result["psr"] + 1e-9

    def test_deterministic(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0.01, 0.03, size=48)
        r1 = deflated_sharpe_ratio(returns, n_trials=8)
        r2 = deflated_sharpe_ratio(returns, n_trials=8)
        assert r1 == r2

    def test_sharpe_is_annualized_with_ddof_1(self):
        returns = np.array([0.01, 0.02, 0.01, 0.02, 0.01, 0.02] * 5)
        result = deflated_sharpe_ratio(returns, n_trials=1, periods_per_year=12)
        mean = returns.mean()
        std = returns.std(ddof=1)
        expected_sharpe = (mean / std) * np.sqrt(12)
        assert result["sharpe"] == pytest.approx(expected_sharpe)

    def test_n_trials_one_no_inflation(self):
        # with a single trial there's no multiple-testing correction, so
        # DSR should equal PSR (SR_benchmark stays 0)
        rng = np.random.default_rng(11)
        returns = rng.normal(0.01, 0.02, size=40)
        result = deflated_sharpe_ratio(returns, n_trials=1, periods_per_year=12)
        assert result["dsr"] == pytest.approx(result["psr"])

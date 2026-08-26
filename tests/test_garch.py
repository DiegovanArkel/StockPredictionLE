"""Tests for stockpred.models.garch: GJR-GARCH(1,1,1) skew-t simulation
layer with empirical bootstrap fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.models.garch import GarchResult, fit_and_simulate, garch_all

# Small n_sims everywhere for test speed (brief allows this).
_N_SIMS = 500


def _simulate_gjr_garch(
    n: int,
    omega: float = 2e-6,
    alpha: float = 0.08,
    gamma: float = 0.05,
    beta: float = 0.88,
    seed: int = 123,
) -> tuple[np.ndarray, float]:
    """Simulate a GJR-GARCH(1,1,1) daily-return series with normal
    innovations. Returns (returns, true_unconditional_annualized_vol).

    Persistence alpha + gamma/2 + beta must be < 1 for stationarity; the
    defaults give ~0.985, a realistic equity-like persistence.
    """
    persistence = alpha + gamma / 2.0 + beta
    assert persistence < 1.0, "GJR-GARCH parameters must be stationary"

    rng = np.random.default_rng(seed)
    eps = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1.0 - persistence)
    eps[0] = np.sqrt(sigma2[0]) * rng.standard_normal()

    for t in range(1, n):
        indicator = 1.0 if eps[t - 1] < 0 else 0.0
        sigma2[t] = (
            omega
            + alpha * eps[t - 1] ** 2
            + gamma * indicator * eps[t - 1] ** 2
            + beta * sigma2[t - 1]
        )
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()

    true_ann_vol = float(np.sqrt(omega / (1.0 - persistence)) * np.sqrt(252))
    return eps, true_ann_vol


def _assert_monotone_quantiles(quantiles: dict[str, float]) -> None:
    ordered = [quantiles[k] for k in ["q05", "q25", "q50", "q75", "q95"]]
    assert ordered == sorted(ordered)


class TestFitAndSimulateGarchRecovery:
    def test_recovers_ann_vol_within_factor_two(self):
        returns, true_ann_vol = _simulate_gjr_garch(n=2000)
        result = fit_and_simulate(pd.Series(returns), n_sims=_N_SIMS, seed=0)

        assert result.converged is True
        assert true_ann_vol / 2.0 <= result.ann_vol_forecast <= true_ann_vol * 2.0

    def test_returns_garch_result_dataclass(self):
        returns, _ = _simulate_gjr_garch(n=2000)
        result = fit_and_simulate(pd.Series(returns), n_sims=_N_SIMS, seed=0)
        assert isinstance(result, GarchResult)
        assert set(result.month_quantiles) == {"q05", "q25", "q50", "q75", "q95"}
        assert set(result.year_quantiles) == {"q05", "q25", "q50", "q75", "q95"}


class TestBootstrapFallback:
    def test_fallback_triggers_below_500_obs(self):
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(scale=0.01, size=100))
        result = fit_and_simulate(returns, n_sims=_N_SIMS, seed=0)
        assert result.converged is False

    def test_fallback_quantiles_populated(self):
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(scale=0.01, size=100))
        result = fit_and_simulate(returns, n_sims=_N_SIMS, seed=0)
        assert set(result.month_quantiles) == {"q05", "q25", "q50", "q75", "q95"}
        assert set(result.year_quantiles) == {"q05", "q25", "q50", "q75", "q95"}
        assert result.ann_vol_forecast > 0

    def test_fallback_drops_nan_before_counting_obs(self):
        rng = np.random.default_rng(1)
        # 600 real observations padded with NaNs -- after dropna there are
        # still >= 500 obs, so this should NOT trip the <500 fallback purely
        # due to raw length.
        values = rng.normal(scale=0.01, size=600)
        returns = pd.Series(np.concatenate([values, [np.nan] * 50]))
        result = fit_and_simulate(returns, n_sims=_N_SIMS, seed=0)
        # 600 clean obs is enough to attempt a real GARCH fit (which should
        # converge on well-behaved synthetic data).
        assert result.converged is True


class TestEmptyInput:
    def test_all_nan_series_raises_value_error(self):
        with pytest.raises(ValueError):
            fit_and_simulate(pd.Series([np.nan] * 10), n_sims=_N_SIMS, seed=0)


class TestQuantileMonotonicity:
    def test_monotone_on_garch_fit(self):
        returns, _ = _simulate_gjr_garch(n=2000)
        result = fit_and_simulate(pd.Series(returns), n_sims=_N_SIMS, seed=0)
        _assert_monotone_quantiles(result.month_quantiles)
        _assert_monotone_quantiles(result.year_quantiles)

    def test_monotone_on_bootstrap_fallback(self):
        rng = np.random.default_rng(2)
        returns = pd.Series(rng.normal(scale=0.01, size=100))
        result = fit_and_simulate(returns, n_sims=_N_SIMS, seed=0)
        _assert_monotone_quantiles(result.month_quantiles)
        _assert_monotone_quantiles(result.year_quantiles)


class TestDeterminism:
    def test_same_seed_same_result_garch_fit(self):
        returns, _ = _simulate_gjr_garch(n=2000)
        r1 = fit_and_simulate(pd.Series(returns), n_sims=_N_SIMS, seed=7)
        r2 = fit_and_simulate(pd.Series(returns), n_sims=_N_SIMS, seed=7)
        assert r1 == r2

    def test_different_seed_different_result_garch_fit(self):
        returns, _ = _simulate_gjr_garch(n=2000)
        r1 = fit_and_simulate(pd.Series(returns), n_sims=_N_SIMS, seed=7)
        r2 = fit_and_simulate(pd.Series(returns), n_sims=_N_SIMS, seed=8)
        assert r1.month_quantiles != r2.month_quantiles

    def test_same_seed_same_result_bootstrap(self):
        rng = np.random.default_rng(3)
        returns = pd.Series(rng.normal(scale=0.01, size=100))
        r1 = fit_and_simulate(returns, n_sims=_N_SIMS, seed=7)
        r2 = fit_and_simulate(returns, n_sims=_N_SIMS, seed=7)
        assert r1 == r2


class TestGarchAll:
    def _make_prices(self) -> pd.DataFrame:
        dates = pd.bdate_range("2015-01-01", periods=2200)
        frames = []
        for ticker, seed in [("AAA", 10), ("BBB", 11), ("CCC_SHORT", 12)]:
            n = len(dates) if ticker != "CCC_SHORT" else 200
            tdates = dates[-n:]
            eps, _ = _simulate_gjr_garch(n=n, seed=seed)
            price = 100.0 * np.cumprod(1.0 + eps)
            frames.append(
                pd.DataFrame(
                    {
                        "date": tdates,
                        "ticker": ticker,
                        "close": price,
                        "adj_close": price,
                        "volume": 1_000_000,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)

    def test_schema_and_one_row_per_ticker(self):
        prices = self._make_prices()
        out = garch_all(prices, lookback_years=8)

        expected_cols = [
            "ticker",
            "ann_vol",
            "m_q05",
            "m_q25",
            "m_q50",
            "m_q75",
            "m_q95",
            "y_q05",
            "y_q25",
            "y_q50",
            "y_q75",
            "y_q95",
            "converged",
        ]
        assert list(out.columns) == expected_cols
        assert set(out["ticker"]) == {"AAA", "BBB", "CCC_SHORT"}
        assert len(out) == 3

        # The long series should hit the real GARCH fit; the short one
        # (200 obs) should fall back to bootstrap.
        short_row = out.loc[out["ticker"] == "CCC_SHORT"].iloc[0]
        assert short_row["converged"] == False  # noqa: E712

    def test_skips_ticker_with_no_usable_history(self):
        prices = self._make_prices()
        # A ticker with a single row -> pct_change() yields zero non-NaN
        # observations -> fit_and_simulate raises -> garch_all should log
        # and skip it, not crash.
        one_row = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-02")],
                "ticker": ["DDD_NEW"],
                "close": [50.0],
                "adj_close": [50.0],
                "volume": [1000],
            }
        )
        prices_with_new = pd.concat([prices, one_row], ignore_index=True)

        out = garch_all(prices_with_new, lookback_years=8)
        assert "DDD_NEW" not in set(out["ticker"])
        assert set(out["ticker"]) == {"AAA", "BBB", "CCC_SHORT"}

    def test_deterministic_regardless_of_input_order(self):
        prices = self._make_prices()
        shuffled = prices.sample(frac=1.0, random_state=0).reset_index(drop=True)

        out1 = garch_all(prices, lookback_years=8).sort_values("ticker").reset_index(drop=True)
        out2 = garch_all(shuffled, lookback_years=8).sort_values("ticker").reset_index(drop=True)

        pd.testing.assert_frame_equal(out1, out2)

    def test_skips_ticker_that_raises_unexpectedly(self, monkeypatch):
        import stockpred.models.garch as garch_mod

        prices = self._make_prices()
        real_fit = garch_mod.fit_and_simulate
        calls = {"n": 0}

        def _flaky(daily_returns, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return real_fit(daily_returns, *args, **kwargs)

        monkeypatch.setattr(garch_mod, "fit_and_simulate", _flaky)
        out = garch_mod.garch_all(prices, lookback_years=8)

        # One ticker's call raised and was skipped; the other two still
        # produced rows.
        assert len(out) == 2
        assert set(out["ticker"]) <= {"AAA", "BBB", "CCC_SHORT"}

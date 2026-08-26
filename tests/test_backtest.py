"""Tests for stockpred.backtest: threshold decision-rule backtest with
deflated Sharpe (Task 9).

Conventions mirror tests/test_conformal.py and tests/test_artifacts.py:
small synthetic frames with hand-computed expected values, returns as
decimal fractions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockpred.backtest import run_backtest
from stockpred.config import Config


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


def _row(ticker, date, q50, q05_cal, y_true, fold=0):
    return {
        "ticker": ticker,
        "date": pd.Timestamp(date),
        "q50": q50,
        "q05_cal": q05_cal,
        "y_true": y_true,
        "fold": fold,
    }


class TestZeroCostAllPass:
    def test_net_return_equals_equal_weight_mean(self):
        # Every ticker passes the signal every month; zero trading costs ->
        # net return must equal the plain equal-weight mean of y_true.
        cfg = _make_config(
            cost_bps=0.0,
            cost_fixed_eur=0.0,
            signal_threshold=-1.0,
            loss_tolerance=-1.0,
        )
        rows = [
            *[_row(t, "2020-01-31", 0.05, 0.0, y, fold=0) for t, y in [("A", 0.10), ("B", 0.02), ("C", -0.01)]],
            *[_row(t, "2020-02-28", 0.05, 0.0, y, fold=0) for t, y in [("A", -0.03), ("B", 0.04), ("C", 0.01)]],
        ]
        df = pd.DataFrame(rows)
        result = run_backtest(df, cfg)

        expected_jan = np.mean([0.10, 0.02, -0.01])
        expected_feb = np.mean([-0.03, 0.04, 0.01])

        assert result["monthly_returns"][0]["ret"] == pytest.approx(expected_jan)
        assert result["monthly_returns"][1]["ret"] == pytest.approx(expected_feb)
        assert result["monthly_returns"][0]["gross_ret"] == pytest.approx(expected_jan)
        assert result["monthly_returns"][0]["cost_eur"] == 0.0
        assert result["monthly_returns"][0]["n_positions"] == 3

        expected_total = (1 + expected_jan) * (1 + expected_feb) - 1
        assert result["total_return"] == pytest.approx(expected_total)

        # All tickers pass every month, so the benchmark (all tickers,
        # equal-weight, no cost) coincides with the zero-cost portfolio.
        assert result["benchmark_total_return"] == pytest.approx(expected_total)


class TestCostsReduceReturn:
    def test_net_below_gross_when_turnover_and_costs_present(self):
        cfg = _make_config(
            cost_bps=50.0,
            cost_fixed_eur=5.0,
            signal_threshold=-1.0,
            loss_tolerance=-1.0,
        )
        rows = [
            _row("A", "2020-01-31", 0.05, 0.0, 0.10, fold=0),
            _row("B", "2020-01-31", 0.05, 0.0, 0.10, fold=0),
        ]
        df = pd.DataFrame(rows)
        result = run_backtest(df, cfg)

        m = result["monthly_returns"][0]
        assert m["gross_ret"] == pytest.approx(0.10)
        assert m["cost_eur"] > 0.0
        assert m["ret"] < m["gross_ret"]

    def test_zero_turnover_second_month_charges_only_terminal_liquidation(self):
        # Same two tickers held both months -> no entries/exits in month 2,
        # so month 2 carries NO turnover cost. It is the last month though,
        # and the backtest ends holding both tickers, so it does carry the
        # terminal liquidation cost -- and exactly that, nothing more.
        cfg = _make_config(
            cost_bps=50.0,
            cost_fixed_eur=5.0,
            signal_threshold=-1.0,
            loss_tolerance=-1.0,
        )
        rows = [
            _row("A", "2020-01-31", 0.05, 0.0, 0.10, fold=0),
            _row("B", "2020-01-31", 0.05, 0.0, 0.10, fold=0),
            _row("A", "2020-02-29", 0.05, 0.0, 0.05, fold=0),
            _row("B", "2020-02-29", 0.05, 0.0, 0.05, fold=0),
        ]
        df = pd.DataFrame(rows)
        result = run_backtest(df, cfg)

        m1, m2 = result["monthly_returns"]

        # Reconstruct the expected terminal liquidation exactly.
        capital_after_m1 = 10000.0 * (1.0 + m1["ret"])
        capital_before_liquidation = capital_after_m1 * (1.0 + m2["gross_ret"])
        notional_final = capital_before_liquidation / 2
        expected_exit = 2 * (notional_final * cfg.cost_bps / 1e4 + cfg.cost_fixed_eur)

        assert m2["cost_eur"] == pytest.approx(expected_exit)
        assert m2["ret"] == pytest.approx(m2["gross_ret"] - expected_exit / capital_after_m1)
        # And the equity identity still holds.
        assert result["final_capital"] == pytest.approx(
            capital_before_liquidation - expected_exit
        )


class TestNoSignalMonths:
    def test_flat_return_and_zero_cost(self):
        cfg = _make_config(
            cost_bps=50.0,
            cost_fixed_eur=5.0,
            signal_threshold=0.5,  # nothing clears this
            loss_tolerance=-1.0,
        )
        rows = [
            _row("A", "2020-01-31", 0.05, 0.0, 0.10, fold=0),
            _row("B", "2020-01-31", 0.05, 0.0, -0.20, fold=0),
        ]
        df = pd.DataFrame(rows)
        result = run_backtest(df, cfg)

        m = result["monthly_returns"][0]
        assert m["ret"] == 0.0
        assert m["gross_ret"] == 0.0
        assert m["cost_eur"] == 0.0
        assert m["n_positions"] == 0

    def test_loss_tolerance_filter_excludes_ticker(self):
        # q50 clears the threshold but q05_cal is at/below loss_tolerance ->
        # excluded from the selected set.
        cfg = _make_config(
            cost_bps=0.0,
            cost_fixed_eur=0.0,
            signal_threshold=0.01,
            loss_tolerance=-0.10,
        )
        rows = [
            _row("A", "2020-01-31", 0.05, -0.15, 0.10, fold=0),  # excluded
            _row("B", "2020-01-31", 0.05, -0.05, 0.20, fold=0),  # included
        ]
        df = pd.DataFrame(rows)
        result = run_backtest(df, cfg)

        m = result["monthly_returns"][0]
        assert m["n_positions"] == 1
        assert m["gross_ret"] == pytest.approx(0.20)


class TestLiquidationOnNoSignalTransition:
    def test_hold_then_no_signal_then_reenter_charges_exit_cost(self):
        # month1: A, B both selected -> both entered (fresh cost).
        # month2: nothing clears the threshold -> A, B both exit (real
        # liquidation, charged), gross_ret == 0.0, ret == -cost/capital.
        # month3: A selected again -> fresh entry (not a "held" carry-over,
        # since prev_holdings was reset to empty by month2).
        cfg = _make_config(
            cost_bps=50.0,
            cost_fixed_eur=5.0,
            signal_threshold=0.01,
            loss_tolerance=-1.0,
        )
        rows = [
            _row("A", "2020-01-31", 0.05, 0.0, 0.02, fold=0),
            _row("B", "2020-01-31", 0.05, 0.0, 0.03, fold=0),
            _row("A", "2020-02-29", 0.00, 0.0, 0.01, fold=0),  # fails threshold
            _row("B", "2020-02-29", 0.00, 0.0, 0.01, fold=0),  # fails threshold
            _row("A", "2020-03-31", 0.05, 0.0, 0.01, fold=0),
        ]
        df = pd.DataFrame(rows)
        result = run_backtest(df, cfg)

        m1, m2, m3 = result["monthly_returns"]

        assert m1["n_positions"] == 2
        assert m2["n_positions"] == 0
        assert m3["n_positions"] == 1

        # Month 2 is a real liquidation, not a free flat month.
        assert m2["gross_ret"] == 0.0
        assert m2["cost_eur"] > 0.0

        capital_after_m1 = 10000.0 * (1.0 + m1["ret"])
        expected_notional_old = capital_after_m1 / 2  # 2 tickers held going into month 2
        expected_cost_m2 = 2 * (
            expected_notional_old * cfg.cost_bps / 1e4 + cfg.cost_fixed_eur
        )
        assert m2["cost_eur"] == pytest.approx(expected_cost_m2)
        assert m2["ret"] == pytest.approx(-m2["cost_eur"] / capital_after_m1)

        # Month 3's re-entry is charged as a fresh entry (A was fully
        # liquidated in month 2, so this is not a held-through position).
        assert m3["cost_eur"] > 0.0

        # n_trades: 2 entries (m1) + 2 exits (m2 liquidation) + 1 entry (m3)
        # + 1 terminal exit (the backtest ends holding A).
        assert result["n_trades"] == 6


class TestMaxDrawdown:
    def test_exact_on_constructed_path(self):
        # Single ticker each month, zero cost: net returns = [0.20, -0.50].
        # equity = [1.2, 0.6]; running max = [1.2, 1.2];
        # drawdown = [0, -0.5] -> max_drawdown = -0.5 exactly.
        cfg = _make_config(
            cost_bps=0.0,
            cost_fixed_eur=0.0,
            signal_threshold=-1.0,
            loss_tolerance=-1.0,
        )
        rows = [
            _row("A", "2020-01-31", 0.05, 0.0, 0.20, fold=0),
            _row("A", "2020-02-29", 0.05, 0.0, -0.50, fold=0),
        ]
        df = pd.DataFrame(rows)
        result = run_backtest(df, cfg)

        assert result["max_drawdown"] == pytest.approx(-0.5)


class TestNTrades:
    def test_counts_enters_and_exits_on_rotation(self):
        # month1: {A, B} enter (2). month2: {B, C} -> A exits, C enters (2).
        # month3: {A, B} -> C exits, A enters (2). Then the backtest ends
        # holding {A, B}, both of which are liquidated (2). Total = 8.
        cfg = _make_config(
            cost_bps=1.0,
            cost_fixed_eur=0.0,
            signal_threshold=-1.0,
            loss_tolerance=-1.0,
        )
        rows = [
            _row("A", "2020-01-31", 0.05, 0.0, 0.01, fold=0),
            _row("B", "2020-01-31", 0.05, 0.0, 0.01, fold=0),
            _row("B", "2020-02-29", 0.05, 0.0, 0.01, fold=0),
            _row("C", "2020-02-29", 0.05, 0.0, 0.01, fold=0),
            _row("A", "2020-03-31", 0.05, 0.0, 0.01, fold=0),
            _row("B", "2020-03-31", 0.05, 0.0, 0.01, fold=0),
        ]
        df = pd.DataFrame(rows)
        result = run_backtest(df, cfg)

        assert result["n_trades"] == 8


class TestTerminalLiquidation:
    """A backtest that ends holding positions must pay to get out of them --
    ``final_capital`` is cash, not paper (see run_backtest's cost model).
    """

    def test_ending_in_cash_charges_nothing_extra(self):
        cfg = _make_config(
            cost_bps=50.0,
            cost_fixed_eur=5.0,
            signal_threshold=0.01,
            loss_tolerance=-1.0,
        )
        rows = [
            _row("A", "2020-01-31", 0.05, 0.0, 0.02, fold=0),
            _row("A", "2020-02-29", 0.00, 0.0, 0.01, fold=0),  # exits, ends in cash
        ]
        result = run_backtest(pd.DataFrame(rows), cfg)

        m1, m2 = result["monthly_returns"]
        capital_after_m1 = 10000.0 * (1.0 + m1["ret"])
        expected_exit = 1 * (capital_after_m1 / 1 * cfg.cost_bps / 1e4 + cfg.cost_fixed_eur)
        # Month 2's cost is the ordinary signal-driven exit and nothing else.
        assert m2["cost_eur"] == pytest.approx(expected_exit)
        assert result["n_trades"] == 2  # 1 entry + 1 exit, no terminal charge

    def test_terminal_exit_charged_exactly_once_at_final_notional(self):
        cfg = _make_config(
            cost_bps=50.0,
            cost_fixed_eur=5.0,
            signal_threshold=-1.0,
            loss_tolerance=-1.0,
        )
        rows = [_row("A", "2020-01-31", 0.05, 0.0, 0.10, fold=0)]
        result = run_backtest(pd.DataFrame(rows), cfg)

        m = result["monthly_returns"][0]
        entry_cost = 10000.0 / 1 * cfg.cost_bps / 1e4 + cfg.cost_fixed_eur
        # capital_{t+1} = capital_t * (1 + gross_ret - cost / capital_t)
        capital_before_liq = 10000.0 * (1.0 + 0.10 - entry_cost / 10000.0)
        exit_cost = capital_before_liq * cfg.cost_bps / 1e4 + cfg.cost_fixed_eur

        assert m["cost_eur"] == pytest.approx(entry_cost + exit_cost)
        assert result["n_trades"] == 2  # 1 entry + 1 terminal exit
        assert result["final_capital"] == pytest.approx(capital_before_liq - exit_cost)
        assert result["final_capital"] == pytest.approx(
            10000.0 * (1.0 + result["total_return"])
        )

    def test_zero_cost_config_leaves_results_untouched(self):
        cfg = _make_config(
            cost_bps=0.0,
            cost_fixed_eur=0.0,
            signal_threshold=-1.0,
            loss_tolerance=-1.0,
        )
        rows = [
            _row("A", "2020-01-31", 0.05, 0.0, 0.10, fold=0),
            _row("A", "2020-02-29", 0.05, 0.0, 0.05, fold=0),
        ]
        result = run_backtest(pd.DataFrame(rows), cfg)
        assert result["monthly_returns"][-1]["cost_eur"] == 0.0
        assert result["total_return"] == pytest.approx(1.10 * 1.05 - 1.0)


class TestOutputShapeAndSerializability:
    def _synthetic(self, n_months=10, n_tickers=5, seed=0):
        rng = np.random.default_rng(seed)
        rows = []
        months = pd.date_range("2020-01-31", periods=n_months, freq="ME")
        for i, date in enumerate(months):
            fold = i % 3
            for j in range(n_tickers):
                ticker = f"T{j}"
                q50 = rng.normal(scale=0.05)
                q05_cal = q50 - abs(rng.normal(scale=0.05)) - 0.01
                y_true = rng.normal(scale=0.08)
                rows.append(_row(ticker, date, q50, q05_cal, y_true, fold=fold))
        return pd.DataFrame(rows)

    def test_json_serializable_and_keys_present(self):
        cfg = _make_config(n_folds=3)
        df = self._synthetic()
        result = run_backtest(df, cfg)

        expected_keys = {
            "monthly_returns",
            "total_return",
            "ann_return",
            "ann_vol",
            "sharpe",
            "psr",
            "dsr",
            "max_drawdown",
            "n_trades",
            "benchmark_total_return",
            "per_fold_sharpe",
            "n_months",
            "final_capital",
        }
        assert expected_keys.issubset(result.keys())

        # Must round-trip through json with no numpy leakage.
        serialized = json.dumps(result)
        reloaded = json.loads(serialized)
        assert reloaded["n_months"] == 10

        for m in result["monthly_returns"]:
            assert set(m.keys()) == {"date", "ret", "gross_ret", "n_positions", "cost_eur"}
            assert isinstance(m["date"], str)
            assert isinstance(m["ret"], float)
            assert isinstance(m["n_positions"], int)

        assert isinstance(result["n_trades"], int)
        assert isinstance(result["final_capital"], float)
        for pf in result["per_fold_sharpe"]:
            assert set(pf.keys()) == {"fold", "sharpe"}
            assert isinstance(pf["fold"], int)
            assert isinstance(pf["sharpe"], float)

    def test_final_capital_matches_total_return(self):
        cfg = _make_config(n_folds=3)
        df = self._synthetic()
        result = run_backtest(df, cfg)

        expected_final = 10000.0 * (1 + result["total_return"])
        assert result["final_capital"] == pytest.approx(expected_final)

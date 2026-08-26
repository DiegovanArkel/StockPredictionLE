"""Threshold decision-rule backtest with deflated Sharpe (Task 9).

``run_backtest`` turns a calibrated out-of-sample forecast panel (Task 7's
``calibrate_from_wf`` output) into a monthly equal-weight long/cash
portfolio, honestly accounting for trading costs, and reports the deflated
Sharpe ratio (Task 4's :func:`stockpred.validation.deflated_sharpe_ratio`)
alongside plain return/risk statistics.

All returns are decimal fractions (0.05 == 5%), never percent.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from stockpred.validation import deflated_sharpe_ratio

_INITIAL_CAPITAL = 10_000.0


def run_backtest(oos_calibrated: pd.DataFrame, cfg: Any) -> dict:
    """Backtest the threshold decision rule over ``oos_calibrated``.

    ``oos_calibrated`` is Task 7's ``calibrate_from_wf`` output: one row per
    (ticker, date) with columns including ``q50`` (calibrated median),
    ``q05_cal`` (calibrated lower bound), ``y_true`` (next-month realized
    return, decimal), and ``fold`` (the walk-forward fold each row's test
    month belongs to). Each row's decision uses only that row's own
    forecast columns -- no look-ahead is introduced here.

    **Decision rule** (evaluated independently per ticker, per month):
    go long a ticker iff ``q50 > cfg.signal_threshold`` AND
    ``q05_cal > cfg.loss_tolerance``; otherwise stay in cash for that
    ticker. Selected tickers in a given month are held equal-weight; a
    month with no selected tickers is 100% cash for its *gross* return.

    **Capital / cost model** (deterministic, reproducible from the
    ``monthly_returns`` output alone):

    - Capital starts at EUR 10,000 and compounds with each month's *net*
      return: ``capital_{t+1} = capital_t * (1 + ret_t)``.
    - Gross return = equal-weight mean of ``y_true`` over the selected
      tickers, or ``0.0`` when nothing is selected (100% cash).
    - Trading cost is turnover-based and charged only on tickers
      *entering* (in this month's selection, not last month's held set) or
      *exiting* (in last month's held set, not this month's selection) --
      a ticker held in both consecutive months incurs no cost. Liquidating
      into cash is a real exit for every previously-held ticker and is
      costed exactly like any other exit -- a no-signal month is only free
      when it follows another no-signal month (both the held set and the
      selected set are empty, so there is nothing to trade).
    - The two trade directions use different notionals, since they close
      out different capital allocations:

      - *Entered* tickers are costed against **this month's** equal split
        of capital, ``notional_new = capital_t / n_positions_t`` (the size
        of the new position being opened; undefined/unused when nothing is
        entered, which requires ``n_positions_t > 0`` whenever it applies).
      - *Exited* tickers are costed against **last month's** equal split
        of capital, ``notional_old = capital_t / len(prev_holdings)`` (the
        size of the closed position being sold, using this month's starting
        capital but last month's holding count; requires
        ``len(prev_holdings) > 0`` whenever it applies, since ``exited`` is
        only non-empty when there was something held to exit from).

      Each traded ticker's cost is
      ``notional * cfg.cost_bps / 1e4 + cfg.cost_fixed_eur`` using its
      direction's notional above. Total monthly cost is the sum over all
      entered + exited tickers. Net return = gross return -
      total_cost / capital_t. A hold -> no-signal transition therefore has
      ``gross_ret == 0.0`` and ``ret == -cost_eur / capital_t`` (a real,
      negative, capital-compounding cost -- not a free flat month); only a
      cash -> cash transition is truly ``ret == cost_eur == 0.0``.

    - **Terminal liquidation.** If the backtest ends while still holding
      positions, closing them is charged too: ``final_capital`` is
      cash-equivalent, not paper. The exit is priced against the notional
      actually sold -- the end-of-period capital split over the closing
      holdings -- and booked into the **last month's** ``cost_eur``, with
      the last month's ``ret`` reduced by ``exit_cost /
      capital_at_start_of_last_month``. Expressing it as a drag on that
      month's starting capital keeps ``final_capital`` exactly equal to
      ``10_000 * (1 + total_return)`` and keeps the ``monthly_returns``
      series self-consistent (compounding it reproduces the equity curve).
      The closing positions count toward ``n_trades`` as exits. A backtest
      that ends in cash is unaffected.

    **Outputs** -- returns a dict of native Python types only (every value
    round-trips through ``json.dumps``):

    - ``monthly_returns``: list of ``{"date": ISO str, "ret": float,
      "gross_ret": float, "n_positions": int, "cost_eur": float}``, one
      entry per sorted unique date.
    - ``total_return``: compounded net return over the full period.
    - ``ann_return``: ``(1 + total_return) ** (12 / n_months) - 1``.
    - ``ann_vol``: ``std(net_returns, ddof=1) * sqrt(12)`` (0.0 if fewer
      than 2 months).
    - ``sharpe``, ``psr``, ``dsr``: from
      ``deflated_sharpe_ratio(net_returns, n_trials=cfg.n_folds,
      periods_per_year=12)`` -- ``n_trials=cfg.n_folds`` honestly accounts
      for having selected this decision rule after seeing all folds.
    - ``max_drawdown``: ``min(equity / cummax(equity) - 1)`` over the net
      return equity curve (0.0 with no months).
    - ``n_trades``: total count of entry + exit events across all months.
    - ``benchmark_total_return``: compounded return of an equal-weight,
      no-cost, buy-and-hold-everything portfolio (every ticker present
      that month, regardless of signal).
    - ``per_fold_sharpe``: list of ``{"fold": int, "sharpe": float}``, one
      per distinct ``fold`` value, computed from that fold's own months'
      net returns (the ``sharpe`` key of ``deflated_sharpe_ratio`` does not
      depend on ``n_trials``, so ``n_trials=1`` is used there).
    - ``n_months``: number of unique months backtested.
    - ``final_capital``: capital in EUR after the full period.
    """
    df = oos_calibrated.copy()
    df["date"] = pd.to_datetime(df["date"])
    unique_dates = sorted(df["date"].unique())

    capital = _INITIAL_CAPITAL
    prev_holdings: set = set()
    n_trades = 0
    last_month_start_capital = _INITIAL_CAPITAL
    last_fold: int | None = None

    monthly_returns: list[dict] = []
    benchmark_returns: list[float] = []
    per_fold_returns: dict[int, list[float]] = defaultdict(list)

    for date in unique_dates:
        month_df = df[df["date"] == date]

        bench_ret = float(month_df["y_true"].mean()) if len(month_df) else 0.0
        benchmark_returns.append(bench_ret)

        fold = int(month_df["fold"].iloc[0])

        selected = month_df[
            (month_df["q50"] > cfg.signal_threshold)
            & (month_df["q05_cal"] > cfg.loss_tolerance)
        ]
        selected_tickers = set(selected["ticker"])
        n_positions = len(selected_tickers)

        gross_ret = float(selected["y_true"].mean()) if n_positions else 0.0

        # Entered/exited are computed uniformly regardless of whether this
        # month has any selections at all -- a no-signal month with
        # non-empty prev_holdings is a real liquidation (every held ticker
        # exits), not a costless no-op; only a cash->cash month (both sets
        # empty) is genuinely free.
        entered = selected_tickers - prev_holdings
        exited = prev_holdings - selected_tickers

        cost_eur = 0.0
        if entered:
            # New positions are costed against *this* month's equal split
            # of capital (entered is non-empty => n_positions > 0, so this
            # division is safe).
            notional_new = capital / n_positions
            cost_eur += len(entered) * (notional_new * cfg.cost_bps / 1e4 + cfg.cost_fixed_eur)
        if exited:
            # Closed positions are costed against the notional they
            # actually held -- last month's equal split of capital over
            # last month's holding count (exited is non-empty =>
            # prev_holdings is non-empty, so this division is safe) -- not
            # this month's (possibly zero) position count.
            notional_old = capital / len(prev_holdings)
            cost_eur += len(exited) * (notional_old * cfg.cost_bps / 1e4 + cfg.cost_fixed_eur)

        n_traded = len(entered) + len(exited)
        n_trades += n_traded

        ret = gross_ret - cost_eur / capital
        prev_holdings = selected_tickers

        last_month_start_capital = capital
        last_fold = fold
        capital = capital * (1.0 + ret)

        monthly_returns.append(
            {
                "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                "ret": float(ret),
                "gross_ret": float(gross_ret),
                "n_positions": int(n_positions),
                "cost_eur": float(cost_eur),
            }
        )
        per_fold_returns[fold].append(ret)

    # --- Terminal liquidation -------------------------------------------
    # The backtest ends holding positions, and those positions are not free
    # to be rid of: an investor comparing this to a benchmark can only spend
    # cash. Charge the exit cost of everything still held at the end,
    # priced against the final capital split over the closing holdings --
    # the notional actually being sold. Booked into the LAST month (it
    # happens at that month's close), expressed as a return drag relative
    # to that month's starting capital so the reported monthly return
    # series still compounds exactly to `final_capital`.
    if prev_holdings and monthly_returns:
        notional_final = capital / len(prev_holdings)
        exit_cost = len(prev_holdings) * (
            notional_final * cfg.cost_bps / 1e4 + cfg.cost_fixed_eur
        )
        n_trades += len(prev_holdings)
        capital -= exit_cost
        drag = exit_cost / last_month_start_capital
        monthly_returns[-1]["ret"] = float(monthly_returns[-1]["ret"] - drag)
        monthly_returns[-1]["cost_eur"] = float(monthly_returns[-1]["cost_eur"] + exit_cost)
        if last_fold is not None:
            per_fold_returns[last_fold][-1] -= drag

    net_returns = np.array([m["ret"] for m in monthly_returns], dtype=float)
    n_months = len(net_returns)

    total_return = float(np.prod(1.0 + net_returns) - 1.0) if n_months else 0.0
    ann_return = float((1.0 + total_return) ** (12.0 / n_months) - 1.0) if n_months else 0.0
    ann_vol = float(net_returns.std(ddof=1) * np.sqrt(12.0)) if n_months >= 2 else 0.0

    dsr_result = deflated_sharpe_ratio(net_returns, n_trials=cfg.n_folds, periods_per_year=12)

    if n_months:
        equity = np.cumprod(1.0 + net_returns)
        running_max = np.maximum.accumulate(equity)
        drawdown = equity / running_max - 1.0
        max_drawdown = float(drawdown.min())
    else:
        max_drawdown = 0.0

    benchmark_total_return = (
        float(np.prod(1.0 + np.array(benchmark_returns, dtype=float)) - 1.0)
        if benchmark_returns
        else 0.0
    )

    per_fold_sharpe = []
    for fold_id in sorted(per_fold_returns.keys()):
        fold_returns = np.array(per_fold_returns[fold_id], dtype=float)
        fold_sharpe = deflated_sharpe_ratio(fold_returns, n_trials=1, periods_per_year=12)["sharpe"]
        per_fold_sharpe.append({"fold": int(fold_id), "sharpe": float(fold_sharpe)})

    return {
        "monthly_returns": monthly_returns,
        "total_return": total_return,
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": dsr_result["sharpe"],
        "psr": dsr_result["psr"],
        "dsr": dsr_result["dsr"],
        "max_drawdown": max_drawdown,
        "n_trades": int(n_trades),
        "benchmark_total_return": benchmark_total_return,
        "per_fold_sharpe": per_fold_sharpe,
        "n_months": int(n_months),
        "final_capital": float(capital),
    }

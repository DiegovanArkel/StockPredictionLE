"""Tests for stockpred.features.stock.build_monthly_panel.

Synthetic data is built with fully known, hand-computable structure so the
momentum/vol/no-lookahead math can be checked exactly (or via an
independently-derived expected value), not just "runs without error".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.features import stock

TRADING_DAYS_PER_YEAR = 252
DAYS_PER_MONTH = 5  # synthetic trading days per month


def _month_end_dates(n_months: int, start: str = "2018-01-31") -> list[pd.Timestamp]:
    """n_months consecutive calendar month-end timestamps starting at `start`."""
    first = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
    return [first + pd.offsets.MonthEnd(i) for i in range(n_months)]


def _make_ticker_prices(
    ticker: str,
    monthly_returns: list[float],
    start_price: float = 100.0,
    days_per_month: int = DAYS_PER_MONTH,
    volume: float = 1_000.0,
    start_month_end: str = "2018-01-31",
) -> pd.DataFrame:
    """Build daily prices for one ticker with a "warm-up" month 0 plus one row
    per `monthly_returns` entry, where month i (1-indexed among the listed
    returns) has adj_close-month-end level = level[i-1] * (1+monthly_returns[i-1]).

    Within each month, all `days_per_month` daily returns are IDENTICAL
    (geometric mean of the month's total return) so vol_1m has a known,
    exact answer (0, since all daily returns in a month equal the constant
    daily rate `g`).
    """
    month_ends = _month_end_dates(1 + len(monthly_returns), start=start_month_end)
    rows = []
    level = start_price
    for month_idx, month_end in enumerate(month_ends):
        if month_idx == 0:
            month_ret = 0.0  # warm-up month: no true return, just seed prices
        else:
            month_ret = monthly_returns[month_idx - 1]
        target_level = level * (1.0 + month_ret)
        g = (target_level / level) ** (1.0 / days_per_month) - 1.0
        # business days within this month ending exactly at month_end
        dates = pd.bdate_range(end=month_end, periods=days_per_month)
        for day_idx, d in enumerate(dates, start=1):
            level = level * (1.0 + g) if not (month_idx == 0 and day_idx == 1) else level
            rows.append({"date": d, "close": level, "adj_close": level, "volume": volume})
        # ensure exact target level at month end (avoid float drift)
        rows[-1]["close"] = target_level
        rows[-1]["adj_close"] = target_level
        level = target_level

    df = pd.DataFrame(rows)
    df["ticker"] = ticker
    return df[["date", "ticker", "close", "adj_close", "volume"]]


def _make_factors(n_months: int, mkt_rf: list[float], start_month_end: str = "2018-01-31") -> pd.DataFrame:
    month_ends = _month_end_dates(n_months, start=start_month_end)
    assert len(mkt_rf) == n_months
    return pd.DataFrame(
        {
            "date": month_ends,
            "mkt_rf": mkt_rf,
            "smb": [0.0] * n_months,
            "hml": [0.0] * n_months,
            "mom": [0.0] * n_months,
            "rf": [0.0] * n_months,
        }
    )


def _panel_from_returns(ticker: str, returns: list[float], start_price: float = 100.0) -> pd.DataFrame:
    """Build a monthly panel where `panel.iloc[i]["ret_1m"] == returns[i]`
    exactly, for every i in range(len(returns)).

    `_make_ticker_prices` always prepends a NaN-return warm-up month, and
    `build_monthly_panel` always drops the true last month (its target is
    undefined). Both would otherwise break a naive 1:1 mapping from
    `returns` to panel rows, so: append a filler month (so the last real
    month keeps a defined target and survives the drop), then trim the
    warm-up row away here.
    """
    padded_returns = list(returns) + [0.0]
    prices = _make_ticker_prices(ticker, padded_returns, start_price=start_price)
    n_months = 1 + len(padded_returns)
    factors = _make_factors(n_months, [0.0] * n_months)
    panel = stock.build_monthly_panel(prices, factors).sort_values("date").reset_index(drop=True)
    return panel.iloc[1:].reset_index(drop=True)


def _panel_from_returns_with_mkt(
    ticker: str, returns: list[float], mkt_rf: list[float], start_price: float = 100.0
) -> pd.DataFrame:
    """Like `_panel_from_returns`, but also aligns `mkt_rf[i]` to
    `panel.iloc[i]` exactly (for the beta_12m tests)."""
    assert len(mkt_rf) == len(returns)
    padded_returns = list(returns) + [0.0]
    padded_mkt = [0.0] + list(mkt_rf) + [0.0]  # +1 for warm-up, +1 for filler
    prices = _make_ticker_prices(ticker, padded_returns, start_price=start_price)
    n_months = 1 + len(padded_returns)
    factors = _make_factors(n_months, padded_mkt)
    panel = stock.build_monthly_panel(prices, factors).sort_values("date").reset_index(drop=True)
    return panel.iloc[1:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Column shape / tidy output
# ---------------------------------------------------------------------------

_EXPECTED_COLUMNS = {
    "ticker",
    "date",
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
    "fwd_ret_1m",
}


def test_build_monthly_panel_has_expected_columns_and_tidy_index():
    returns = [0.05, -0.02, 0.03, 0.01, -0.01]
    prices = _make_ticker_prices("AAA", returns)
    factors = _make_factors(6, [0.01] * 6)

    panel = stock.build_monthly_panel(prices, factors)

    assert set(panel.columns) == _EXPECTED_COLUMNS
    assert isinstance(panel.index, pd.RangeIndex)


def test_build_monthly_panel_one_row_per_ticker_month():
    returns = [0.05, -0.02, 0.03]
    prices_a = _make_ticker_prices("AAA", returns)
    prices_b = _make_ticker_prices("BBB", returns)
    prices = pd.concat([prices_a, prices_b], ignore_index=True)
    factors = _make_factors(4, [0.0] * 4)

    panel = stock.build_monthly_panel(prices, factors)

    # 4 months per ticker, minus 1 dropped for missing fwd_ret_1m (last month) = 3
    assert sorted(panel["ticker"].unique()) == ["AAA", "BBB"]
    assert (panel.groupby("ticker").size() == 3).all()
    assert not panel.duplicated(subset=["ticker", "date"]).any()


# ---------------------------------------------------------------------------
# ret_1m / mom_1m exactness
# ---------------------------------------------------------------------------


def test_ret_1m_matches_adj_close_month_end_ratio():
    returns = [0.05, -0.02, 0.03, 0.10, -0.04]
    prices = _make_ticker_prices("AAA", returns)
    factors = _make_factors(6, [0.0] * 6)

    panel = stock.build_monthly_panel(prices, factors).sort_values("date").reset_index(drop=True)

    # panel drops the last month (missing target), so we see months 1..4 of
    # the 5 listed returns (month 0 is the warm-up, dropped: its ret_1m is NaN
    # only in the pre-drop frame, but NaN target is what's dropped, not NaN
    # feature -- month 0's row IS present here since its target is month 1's
    # return, which exists).
    for i, expected_ret in enumerate(returns[:-1]):
        row = panel.iloc[i + 1]  # +1 to skip the warm-up month (index 0)
        assert row["ret_1m"] == pytest.approx(expected_ret, abs=1e-9)
        assert row["mom_1m"] == pytest.approx(expected_ret, abs=1e-9)


# ---------------------------------------------------------------------------
# Momentum: exact math on constructed series
# ---------------------------------------------------------------------------


def test_momentum_windows_exact_on_constructed_series():
    returns = [0.05, -0.02, 0.03, 0.01, -0.01, 0.04, 0.02, -0.03, 0.06, 0.00, 0.01, -0.02, 0.03, 0.02]
    panel = _panel_from_returns("AAA", returns)

    # panel.iloc[i]["ret_1m"] == returns[i] exactly (see _panel_from_returns).
    def expected_cumret(window_returns: list[float]) -> float:
        prod = 1.0
        for r in window_returns:
            prod *= 1.0 + r
        return prod - 1.0

    for i in range(len(returns)):
        row = panel.iloc[i]
        assert row["ret_1m"] == pytest.approx(returns[i], abs=1e-9)

        if i >= 2:
            assert row["mom_3m"] == pytest.approx(expected_cumret(returns[i - 2 : i + 1]), abs=1e-9)
        else:
            assert pd.isna(row["mom_3m"])

        if i >= 5:
            assert row["mom_6m"] == pytest.approx(expected_cumret(returns[i - 5 : i + 1]), abs=1e-9)
        else:
            assert pd.isna(row["mom_6m"])

        if i >= 11:
            assert row["mom_12m"] == pytest.approx(expected_cumret(returns[i - 11 : i + 1]), abs=1e-9)
            assert row["mom_12_1"] == pytest.approx(expected_cumret(returns[i - 11 : i]), abs=1e-9)
        else:
            assert pd.isna(row["mom_12m"])
            assert pd.isna(row["mom_12_1"])


def test_mom_12_1_excludes_current_month_return():
    """mom_12_1 must NOT move when only the current month's return changes."""
    base_returns = [0.01] * 11 + [0.02]  # 12 listed months; last one varies
    panel_a = _panel_from_returns("AAA", base_returns)

    varied_returns = [0.01] * 11 + [0.50]  # huge current-month outlier
    panel_b = _panel_from_returns("AAA", varied_returns)

    idx = 11  # the 12th listed month -- first index where mom_12_1 is defined
    mom_12_1_a = panel_a.iloc[idx]["mom_12_1"]
    mom_12_1_b = panel_b.iloc[idx]["mom_12_1"]
    assert mom_12_1_a == pytest.approx(mom_12_1_b, abs=1e-9)

    # but mom_12m (which includes the current month) DOES move
    assert panel_a.iloc[idx]["mom_12m"] != pytest.approx(panel_b.iloc[idx]["mom_12m"])


# ---------------------------------------------------------------------------
# No-lookahead: the most important tests in this project
# ---------------------------------------------------------------------------


def test_fwd_ret_1m_equals_next_month_ret_1m_no_lookahead():
    returns = [0.05, -0.02, 0.03, 0.01, -0.01, 0.02]
    prices = _make_ticker_prices("AAA", returns)
    n_months = 1 + len(returns)
    factors = _make_factors(n_months, [0.0] * n_months)

    panel = stock.build_monthly_panel(prices, factors).sort_values("date").reset_index(drop=True)

    lookup = panel.set_index("date")["ret_1m"]
    for _, row in panel.iterrows():
        next_month = row["date"] + pd.offsets.MonthEnd(1)
        if next_month in lookup.index:
            assert row["fwd_ret_1m"] == pytest.approx(lookup.loc[next_month], abs=1e-9)


def test_fwd_ret_1m_drops_last_month_per_ticker():
    returns = [0.05, -0.02, 0.03]
    prices = _make_ticker_prices("AAA", returns)
    n_months = 1 + len(returns)
    factors = _make_factors(n_months, [0.0] * n_months)

    panel = stock.build_monthly_panel(prices, factors)

    last_month = prices["date"].max() + pd.offsets.MonthEnd(0)
    assert last_month not in set(panel["date"])
    assert panel["fwd_ret_1m"].isna().sum() == 0


def test_fwd_ret_1m_no_cross_ticker_leakage():
    """Ticker A's target must come from ticker A's own next month, never B's."""
    returns_a = [0.05, -0.02, 0.03, 0.10]
    returns_b = [0.30, 0.30, 0.30, 0.30]  # very different regime
    prices_a = _make_ticker_prices("AAA", returns_a)
    prices_b = _make_ticker_prices("BBB", returns_b)
    prices = pd.concat([prices_a, prices_b], ignore_index=True)
    n_months = 1 + len(returns_a)
    factors = _make_factors(n_months, [0.0] * n_months)

    panel = stock.build_monthly_panel(prices, factors)

    a_rows = panel[panel["ticker"] == "AAA"].sort_values("date")
    # a_rows has one row per surviving AAA month (warm-up + returns_a[:-1]);
    # fwd_ret_1m of row i is the ret_1m of the NEXT month, i.e. returns_a[i].
    for target, expected in zip(a_rows["fwd_ret_1m"], returns_a):
        assert target == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def test_vol_1m_is_zero_when_daily_returns_constant_within_month():
    # by construction, every day within a synthetic month has identical
    # daily return, so vol_1m (std of daily returns within the month) is 0.
    returns = [0.05, -0.02, 0.03]
    prices = _make_ticker_prices("AAA", returns)
    n_months = 1 + len(returns)
    factors = _make_factors(n_months, [0.0] * n_months)

    panel = stock.build_monthly_panel(prices, factors)

    assert (panel["vol_1m"].dropna().abs() < 1e-6).all()


def test_vol_3m_and_vol_12m_nan_before_min_observations():
    # 3m needs >=15 daily obs (3 months * 5/month), 12m needs >=60.
    returns = [0.01] * 14
    panel = _panel_from_returns("AAA", returns)

    assert pd.isna(panel.iloc[0]["vol_3m"])
    assert pd.isna(panel.iloc[1]["vol_3m"])
    assert panel.iloc[2:]["vol_3m"].notna().all()

    assert panel.iloc[:11]["vol_12m"].isna().all()
    assert panel.iloc[11:]["vol_12m"].notna().all()


def test_vol_annualization_matches_manual_pooled_std():
    """Cross-check vol_3m against an independently-computed pooled std over
    the raw daily returns for the trailing 3-month window."""
    returns = [0.05, -0.02, 0.03, 0.08, -0.05, 0.02]
    prices = _make_ticker_prices("AAA", returns).sort_values("date").reset_index(drop=True)
    n_months = 1 + len(returns)
    factors = _make_factors(n_months, [0.0] * n_months)

    panel = stock.build_monthly_panel(prices, factors).sort_values("date").reset_index(drop=True)

    prices = prices.copy()
    prices["daily_ret"] = prices["adj_close"].pct_change()
    prices["month_end"] = prices["date"] + pd.offsets.MonthEnd(0)
    month_ends = sorted(prices["month_end"].unique())

    # pick month index 4 (5th month overall, well past the 3m warm-up)
    target_month_end = month_ends[4]
    trailing_months = month_ends[2:5]
    window_rets = prices[prices["month_end"].isin(trailing_months)]["daily_ret"].dropna()
    expected_vol = window_rets.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

    row = panel[panel["date"] == target_month_end].iloc[0]
    assert row["vol_3m"] == pytest.approx(expected_vol, rel=1e-6)


# ---------------------------------------------------------------------------
# Amihud illiquidity / dollar volume
# ---------------------------------------------------------------------------


def test_amihud_and_dollar_vol_match_manual_computation():
    returns = [0.05, -0.02, 0.03]
    prices = _make_ticker_prices("AAA", returns, volume=2_000.0).sort_values("date").reset_index(drop=True)
    n_months = 1 + len(returns)
    factors = _make_factors(n_months, [0.0] * n_months)

    panel = stock.build_monthly_panel(prices, factors).sort_values("date").reset_index(drop=True)

    prices = prices.copy()
    prices["daily_ret"] = prices["adj_close"].pct_change()
    prices["month_end"] = prices["date"] + pd.offsets.MonthEnd(0)
    prices["dollar_vol_daily"] = prices["close"] * prices["volume"]
    prices["amihud_daily"] = prices["daily_ret"].abs() / prices["dollar_vol_daily"]

    for month_end, group in prices.groupby("month_end"):
        if month_end not in set(panel["date"]):
            continue
        expected_dollar_vol = np.log(group["dollar_vol_daily"].mean())
        expected_amihud = np.log1p(1e6 * group["amihud_daily"].mean())
        row = panel[panel["date"] == month_end].iloc[0]
        assert row["dollar_vol"] == pytest.approx(expected_dollar_vol, rel=1e-9)
        if not pd.isna(row["amihud"]):
            assert row["amihud"] == pytest.approx(expected_amihud, rel=1e-6)


# ---------------------------------------------------------------------------
# beta_12m
# ---------------------------------------------------------------------------


def test_beta_12m_recovers_known_slope():
    n = 19
    rng = np.random.default_rng(42)
    mkt_rf = list(rng.normal(0.0, 0.03, size=n))
    true_beta = 1.5
    stock_returns = [true_beta * m for m in mkt_rf]

    panel = _panel_from_returns_with_mkt("AAA", stock_returns, mkt_rf)

    # once we have >=12 months of (ret_1m, mkt_rf) pairs, beta should recover
    # true_beta closely (noiseless linear relationship)
    last_row = panel.iloc[-1]
    assert last_row["beta_12m"] == pytest.approx(true_beta, abs=1e-6)


def test_beta_12m_nan_before_min_periods():
    n = 5
    rng = np.random.default_rng(7)
    mkt_rf = list(rng.normal(0.0, 0.02, size=n))
    stock_returns = [0.5 * m for m in mkt_rf]

    panel = _panel_from_returns_with_mkt("AAA", stock_returns, mkt_rf)

    # fewer than 12 months of history everywhere -> always NaN
    assert panel["beta_12m"].isna().all()


def test_beta_12m_uses_only_data_up_to_t_no_lookahead():
    """Changing a LATER month's mkt_rf/return must not change beta_12m at an
    earlier month t (rolling window only looks backward)."""
    n = 20
    rng = np.random.default_rng(3)
    mkt_rf = list(rng.normal(0.0, 0.03, size=n))
    true_beta = 0.8
    stock_returns = [true_beta * m for m in mkt_rf]

    panel_a = _panel_from_returns_with_mkt("AAA", stock_returns, mkt_rf)

    # perturb an INTERMEDIATE month with a value inconsistent with true_beta
    # (breaks linearity for any window containing it) -- not the last month,
    # so the perturbed row isn't the one dropped for lacking a target.
    perturb_idx = 15
    mkt_rf_b = list(mkt_rf)
    mkt_rf_b[perturb_idx] = 5.0
    stock_returns_b = list(stock_returns)
    stock_returns_b[perturb_idx] = -0.5  # inconsistent with true_beta * mkt_rf_b (4.0)

    panel_b = _panel_from_returns_with_mkt("AAA", stock_returns_b, mkt_rf_b)

    # beta at an EARLY row (strictly before the perturbed month) must be identical
    early_idx = 5
    assert panel_a.iloc[early_idx]["date"] == panel_b.iloc[early_idx]["date"]
    a_beta_early = panel_a.iloc[early_idx]["beta_12m"]
    b_beta_early = panel_b.iloc[early_idx]["beta_12m"]
    if pd.isna(a_beta_early) and pd.isna(b_beta_early):
        pass
    else:
        assert a_beta_early == pytest.approx(b_beta_early, abs=1e-9)

    # sanity: the perturbation DOES change a later row's beta (within the
    # trailing 36m window) -- otherwise this test would trivially pass
    later_idx = 18
    a_beta_later = panel_a.iloc[later_idx]["beta_12m"]
    b_beta_later = panel_b.iloc[later_idx]["beta_12m"]
    assert a_beta_later != pytest.approx(b_beta_later, abs=1e-6)


# ---------------------------------------------------------------------------
# Mid-series calendar gap (regression: a missing month must never make
# row-position-based rolling/shift silently span the wrong calendar window)
# ---------------------------------------------------------------------------


def test_build_monthly_panel_handles_mid_series_gap_month():
    """A ticker with an entire missing calendar month (e.g. no trading data
    for May at all) must not let neighboring months' features/target be
    computed as if the gap month didn't exist.

    Regression for: April's fwd_ret_1m silently becoming June's return
    (wrong horizon, no NaN) and June's mom_3m silently spanning
    Mar+Apr+Jun instead of the true Apr+May+Jun calendar window.
    """
    ticker = "GAP"
    jan, feb, mar, apr, may, jun, jul = _month_end_dates(7)

    rows: list[dict] = []
    level = 100.0

    def add_month(month_end: pd.Timestamp, month_ret: float, is_first: bool = False) -> None:
        nonlocal level
        target_level = level * (1.0 + month_ret)
        g = (target_level / level) ** (1.0 / DAYS_PER_MONTH) - 1.0
        dates = pd.bdate_range(end=month_end, periods=DAYS_PER_MONTH)
        for day_idx, d in enumerate(dates, start=1):
            if not (is_first and day_idx == 1):
                level = level * (1.0 + g)
            rows.append({"date": d, "close": level, "adj_close": level, "volume": 1_000.0})
        rows[-1]["close"] = target_level
        rows[-1]["adj_close"] = target_level
        level = target_level

    add_month(jan, 0.0, is_first=True)
    add_month(feb, 0.01)
    add_month(mar, 0.01)
    add_month(apr, 0.01)
    # May: NO trading data at all -- the gap.
    add_month(jun, -0.50)
    add_month(jul, 0.02)

    prices = pd.DataFrame(rows)
    prices["ticker"] = ticker
    prices = prices[["date", "ticker", "close", "adj_close", "volume"]]
    factors = _make_factors(7, [0.0] * 7)

    # --- internal pipeline (pre target-drop), to directly check the values
    # the reviewer's regression names: fwd_ret_1m and mom_3m around the gap.
    daily = stock._prepare_daily(prices)
    ticker_daily = daily[daily["ticker"] == ticker].sort_values("date")
    monthly = stock._monthly_aggregate(ticker_daily)
    monthly = stock._reindex_full_calendar(monthly)
    monthly = stock._finalize_price_derived_columns(monthly)
    monthly = stock._add_momentum(monthly)
    monthly["fwd_ret_1m"] = monthly["ret_1m"].shift(-1)

    # May must exist as an explicit (NaN) row internally, so the rolling
    # windows around it stay calendar-aligned.
    assert may in set(monthly["date"])

    # (a) April's fwd_ret_1m must be NaN -- NOT June's -50% return.
    apr_row = monthly[monthly["date"] == apr].iloc[0]
    assert pd.isna(apr_row["fwd_ret_1m"])

    # June's own ret_1m (relative to the unknown May price) is NaN.
    jun_row = monthly[monthly["date"] == jun].iloc[0]
    assert pd.isna(jun_row["ret_1m"])

    # (b) June's mom_3m (true window Apr, May, Jun) must be NaN -- never
    # silently compressed into a Mar+Apr+Jun product.
    assert pd.isna(jun_row["mom_3m"])

    # --- public API: the phantom May row never appears in the output, and
    # April -- whose true target (May's return) is unknown -- is correctly
    # dropped rather than kept with a wrong-horizon target.
    panel = stock.build_monthly_panel(prices, factors).sort_values("date").reset_index(drop=True)
    assert may not in set(panel["date"])
    assert apr not in set(panel["date"])

    # June is a real trading month (has its own price data) and survives,
    # with a well-defined, correctly-computed forward target from July.
    jun_panel_row = panel[panel["date"] == jun].iloc[0]
    assert jun_panel_row["fwd_ret_1m"] == pytest.approx(0.02, abs=1e-9)

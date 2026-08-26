"""Monthly stock feature panel: momentum, volatility, liquidity, beta.

``build_monthly_panel`` is the single entry point: it aggregates tidy daily
``prices`` (see ``stockpred.data.prices``) to month-end, joins ``factors``
(see ``stockpred.data.factors``) for ``mkt_rf``, and derives the LightGBM
training table -- one row per (ticker, month-end date) with features known
strictly as of that month end, plus the ``fwd_ret_1m`` target (next month's
realized return). Rows lacking the target are dropped; feature NaNs are
kept (LightGBM handles missing values natively).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_YEAR = 252
_AMIHUD_SCALE = 1e6

_VOL_WINDOWS: dict[str, tuple[int, int]] = {
    # column name -> (window in months, minimum pooled daily observations)
    "vol_1m": (1, 2),
    "vol_3m": (3, 15),
    "vol_12m": (12, 60),
}
_MOM_WINDOWS: dict[str, int] = {
    # column name -> window in months (cumulative return including month t)
    "mom_3m": 3,
    "mom_6m": 6,
    "mom_12m": 12,
}
_MOM_12_1_WINDOW = 11  # months t-11..t-1, excluding month t

_BETA_WINDOW = 36
_BETA_MIN_PERIODS = 12

_OUTPUT_COLUMNS = [
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
]


def build_monthly_panel(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Build the monthly stock feature panel.

    Parameters
    ----------
    prices:
        Tidy daily frame ``[date, ticker, close, adj_close, volume]``.
    factors:
        Tidy monthly (month-end) frame including ``mkt_rf`` (Fama-French).

    Returns
    -------
    A tidy, reset-index frame with one row per (ticker, month-end date):
    ``ret_1m``, the feature columns listed in the task brief, and the
    ``fwd_ret_1m`` target (``ret_1m`` shifted -1 within ticker). Rows with a
    missing target are dropped; feature NaNs are kept.
    """
    daily = _prepare_daily(prices)
    mkt_rf = factors.set_index("date")["mkt_rf"]

    ticker_frames = []
    for ticker, ticker_daily in daily.groupby("ticker", sort=False):
        ticker_daily = ticker_daily.sort_values("date")
        monthly = _monthly_aggregate(ticker_daily)
        monthly = _reindex_full_calendar(monthly)
        monthly = _finalize_price_derived_columns(monthly)
        monthly = _add_momentum(monthly)
        monthly = _add_volatility(monthly)
        monthly = _add_beta(monthly, mkt_rf)
        monthly["ticker"] = ticker
        monthly["fwd_ret_1m"] = monthly["ret_1m"].shift(-1)
        ticker_frames.append(monthly)

    panel = pd.concat(ticker_frames, ignore_index=True)
    # Drop calendar months that were inserted by _reindex_full_calendar
    # because the ticker had NO real trading data that month (a "phantom"
    # row) -- they exist only so row-position-based rolling/shift above
    # stays aligned to the true calendar across the gap; they were never
    # real (ticker, month) observations themselves.
    panel = panel[panel["had_data"]]
    panel = panel.dropna(subset=["fwd_ret_1m"])
    panel = panel[_OUTPUT_COLUMNS].reset_index(drop=True)
    return panel


def _prepare_daily(prices: pd.DataFrame) -> pd.DataFrame:
    """Sort by ticker/date and derive per-row daily return + liquidity inputs."""
    daily = prices.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    daily["daily_ret"] = daily.groupby("ticker", sort=False)["adj_close"].pct_change()
    daily["month_end"] = daily["date"] + pd.offsets.MonthEnd(0)
    daily["dollar_vol_daily"] = daily["close"] * daily["volume"]
    amihud_daily = daily["daily_ret"].abs() / daily["dollar_vol_daily"]
    daily["amihud_daily"] = amihud_daily.replace([np.inf, -np.inf], np.nan)
    return daily


def _monthly_aggregate(ticker_daily: pd.DataFrame) -> pd.DataFrame:
    """Collapse one ticker's daily rows to one row per REAL trading month
    (months with zero trading days simply have no row here yet -- see
    `_reindex_full_calendar`), keeping the pooled sum/sum-of-squares/count
    of daily returns needed for exact rolling-window volatility later."""
    monthly = (
        ticker_daily.groupby("month_end")
        .agg(
            adj_close_last=("adj_close", "last"),
            dollar_vol_mean=("dollar_vol_daily", "mean"),
            amihud_mean=("amihud_daily", "mean"),
            ret_sum=("daily_ret", "sum"),
            ret_sq_sum=("daily_ret", lambda s: float((s.dropna() ** 2).sum())),
            ret_count=("daily_ret", "count"),
        )
        .reset_index()
        .rename(columns={"month_end": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    return monthly


def _reindex_full_calendar(monthly: pd.DataFrame) -> pd.DataFrame:
    """Reindex a per-ticker monthly frame onto the full month-end calendar
    between its min and max date, inserting an explicit all-NaN row for any
    calendar month with no real trading data.

    Without this, every rolling/shift operation downstream (momentum, vol,
    beta, and the fwd_ret_1m target) is row-POSITION-based and would
    silently treat two months that are actually N calendar months apart as
    adjacent across a data gap -- e.g. assigning April's target the return
    of June (skipping a missing May) instead of NaN. Reindexing makes every
    row correspond to exactly one calendar month, gap or not, so those
    operations are naturally calendar-correct; `had_data` marks which rows
    are real observations (phantom gap rows are dropped from the final
    panel by the caller, but must stay present through the rolling/shift
    steps so neighboring real months see the gap).
    """
    full_range = pd.date_range(monthly["date"].min(), monthly["date"].max(), freq="ME")
    reindexed = (
        monthly.set_index("date").reindex(full_range).rename_axis("date").reset_index()
    )
    reindexed["had_data"] = reindexed["ret_count"].notna()
    return reindexed


def _finalize_price_derived_columns(monthly: pd.DataFrame) -> pd.DataFrame:
    """Derive ret_1m/dollar_vol/amihud AFTER calendar reindexing, so ret_1m's
    pct_change() correctly yields NaN across a gap instead of silently
    compounding the return over the missing month(s)."""
    monthly["ret_1m"] = monthly["adj_close_last"].pct_change()
    monthly["dollar_vol"] = np.log(monthly["dollar_vol_mean"])
    monthly["amihud"] = np.log1p(_AMIHUD_SCALE * monthly["amihud_mean"])
    return monthly


def _add_momentum(monthly: pd.DataFrame) -> pd.DataFrame:
    gross = 1.0 + monthly["ret_1m"]

    monthly["mom_1m"] = monthly["ret_1m"]
    for col, window in _MOM_WINDOWS.items():
        monthly[col] = (
            gross.rolling(window, min_periods=window).apply(np.prod, raw=True) - 1.0
        )

    shifted_gross = gross.shift(1)
    monthly["mom_12_1"] = (
        shifted_gross.rolling(_MOM_12_1_WINDOW, min_periods=_MOM_12_1_WINDOW).apply(
            np.prod, raw=True
        )
        - 1.0
    )
    return monthly


def _add_volatility(monthly: pd.DataFrame) -> pd.DataFrame:
    for col, (window, min_obs) in _VOL_WINDOWS.items():
        monthly[col] = _rolling_pooled_annualized_std(
            monthly["ret_sum"], monthly["ret_sq_sum"], monthly["ret_count"], window, min_obs
        )
    return monthly


def _rolling_pooled_annualized_std(
    sum_s: pd.Series, sumsq_s: pd.Series, count_s: pd.Series, window: int, min_obs: int
) -> pd.Series:
    """Annualized std of daily returns pooled across a trailing `window`-month
    period, computed from monthly (sum, sum-of-squares, count) aggregates so
    it exactly matches the pooled sample std of the underlying daily
    observations (not an average of monthly std's)."""
    sum_roll = sum_s.rolling(window, min_periods=1).sum()
    sumsq_roll = sumsq_s.rolling(window, min_periods=1).sum()
    count_roll = count_s.rolling(window, min_periods=1).sum()

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sum_roll / count_roll
        var = (sumsq_roll - count_roll * mean**2) / (count_roll - 1)
    var = var.clip(lower=0)  # guard tiny negative floats from rounding
    std = np.sqrt(var)
    std = std.where(count_roll >= min_obs)
    return std * np.sqrt(_TRADING_DAYS_PER_YEAR)


def _add_beta(monthly: pd.DataFrame, mkt_rf: pd.Series) -> pd.DataFrame:
    """Rolling regression slope of monthly ``ret_1m`` on ``mkt_rf`` over a
    trailing 36-month window (min 12), using only data up to and including
    each row (pandas rolling is inherently backward-looking)."""
    aligned_mkt_rf = monthly["date"].map(mkt_rf)
    # Mask mkt_rf to NaN wherever ret_1m is NaN so var() is computed over the
    # exact same sample cov() uses internally (pandas' rolling cov already
    # does pairwise-complete-observation NaN exclusion) -- otherwise var's
    # denominator would include extra points cov silently dropped, biasing
    # the slope.
    aligned_mkt_rf_matched = aligned_mkt_rf.where(monthly["ret_1m"].notna())
    cov = monthly["ret_1m"].rolling(_BETA_WINDOW, min_periods=_BETA_MIN_PERIODS).cov(aligned_mkt_rf)
    var = aligned_mkt_rf_matched.rolling(_BETA_WINDOW, min_periods=_BETA_MIN_PERIODS).var()
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = cov / var.replace(0, np.nan)
    monthly["beta_12m"] = beta
    return monthly

"""Price data provider: yfinance (primary and only provider).

Stooq is intentionally not implemented -- its endpoint sits behind a JS
anti-bot challenge that isn't practical to solve here. If yfinance fails for
a ticker, callers fall back to the parquet cache via
``stockpred.data.storage.cached_fetch``.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_TIDY_COLUMNS = ["date", "ticker", "close", "adj_close", "volume"]


def fetch_prices(
    tickers: Sequence[str],
    start: str,
    end: str | None = None,
    yf_suffix: str = ".AS",
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch daily OHLCV for each ticker via yfinance.

    Returns a tidy frame ``[date, ticker, close, adj_close, volume]`` (ticker
    stored WITHOUT the exchange suffix) plus the list of tickers that failed
    to resolve after retries.
    """
    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for ticker in tickers:
        yf_ticker = f"{ticker}{yf_suffix}"
        tidy = _fetch_one(yf_ticker, start, end)
        if tidy is None:
            failed.append(ticker)
            continue
        tidy = tidy.copy()
        tidy["ticker"] = ticker
        frames.append(tidy[_TIDY_COLUMNS])

    if frames:
        result = pd.concat(frames, ignore_index=True)
    else:
        result = pd.DataFrame(columns=_TIDY_COLUMNS)

    return result, failed


def _fetch_one(yf_ticker: str, start: str, end: str | None) -> pd.DataFrame | None:
    """Fetch and normalize one ticker's history, retrying on failure/empty result."""
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            raw = yf.download(
                yf_ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
            )
            if raw is None or raw.empty:
                raise ValueError(f"empty response for {yf_ticker}")
            return _normalize(raw)
        except Exception as exc:  # noqa: BLE001 - any yfinance failure triggers a retry
            logger.warning(
                "attempt %d/%d fetching %s failed: %s", attempt, _MAX_ATTEMPTS, yf_ticker, exc
            )
            if attempt < _MAX_ATTEMPTS:
                time.sleep(2 * attempt)

    logger.warning("%s failed after %d attempts", yf_ticker, _MAX_ATTEMPTS)
    return None


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce a yfinance download frame to tidy [date, close, adj_close, volume]."""
    df = raw.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={"Close": "close", "Adj Close": "adj_close", "Volume": "volume"})

    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "date"

    df = df.reset_index()
    return df[["date", "close", "adj_close", "volume"]]

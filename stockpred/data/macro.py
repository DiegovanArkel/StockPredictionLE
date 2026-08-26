"""Macro data providers: FRED (keyless CSV endpoint) and ECB (ecbdata package).

Both providers are best-effort: a per-series failure is logged and skipped
rather than aborting the whole fetch, since later tasks fall back to the
parquet cache for the dataset as a whole via
``stockpred.data.storage.cached_fetch``.
"""

from __future__ import annotations

import io
import logging
from typing import Sequence

import pandas as pd
import requests
from ecbdata import ecbdata

logger = logging.getLogger(__name__)

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
_USER_AGENT = "stockpred/0.1 (+https://github.com/; data research)"
_FRED_TIMEOUT_SECONDS = 30
_TIDY_COLUMNS = ["date", "series_id", "value"]


def fetch_fred(series_ids: Sequence[str]) -> pd.DataFrame:
    """Fetch each FRED series via the keyless CSV endpoint.

    Returns a tidy frame ``[date, series_id, value]``. A series that fails
    (network error, bad response, all-missing data) is logged and skipped;
    it does not abort the other series.
    """
    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for series_id in series_ids:
        try:
            frames.append(_fetch_one_fred(series_id))
        except Exception as exc:  # noqa: BLE001 - best-effort, one series at a time
            logger.warning("FRED fetch failed for %r: %s", series_id, exc)
            failed.append(series_id)

    if failed:
        logger.warning("FRED series that failed to fetch: %s", failed)

    if frames:
        return pd.concat(frames, ignore_index=True)[_TIDY_COLUMNS]
    return pd.DataFrame(columns=_TIDY_COLUMNS)


def _fetch_one_fred(series_id: str) -> pd.DataFrame:
    url = _FRED_CSV_URL.format(series_id=series_id)
    response = requests.get(
        url, headers={"User-Agent": _USER_AGENT}, timeout=_FRED_TIMEOUT_SECONDS
    )
    response.raise_for_status()

    raw = pd.read_csv(io.StringIO(response.text), na_values=["."])
    if raw.shape[1] < 2:
        raise ValueError(f"unexpected FRED CSV shape for {series_id}: {raw.shape}")

    date_col, value_col = raw.columns[0], raw.columns[1]
    tidy = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col]),
            "series_id": series_id,
            "value": pd.to_numeric(raw[value_col], errors="coerce"),
        }
    )
    return tidy.dropna(subset=["value"])


def fetch_ecb(series_keys: Sequence[str]) -> pd.DataFrame:
    """Fetch each ECB series key via ``ecbdata.ecbdata.get_series``.

    Returns a tidy frame ``[date, series_id, value]``. Each key is fetched
    inside its own try/except -- a failure is logged and that key is
    skipped. If every key fails, an empty frame with the correct columns is
    returned (best-effort by design).
    """
    frames: list[pd.DataFrame] = []

    for series_key in series_keys:
        try:
            raw = ecbdata.get_series(series_key)
            if raw is None or raw.empty:
                raise ValueError(f"empty response for {series_key}")
            tidy = pd.DataFrame(
                {
                    "date": pd.to_datetime(raw["TIME_PERIOD"]),
                    "series_id": series_key,
                    "value": pd.to_numeric(raw["OBS_VALUE"], errors="coerce"),
                }
            ).dropna(subset=["value"])
            frames.append(tidy)
        except Exception as exc:  # noqa: BLE001 - best-effort, one key at a time
            logger.warning("ECB fetch failed for %r: %s", series_key, exc)

    if frames:
        return pd.concat(frames, ignore_index=True)[_TIDY_COLUMNS]
    return pd.DataFrame(columns=_TIDY_COLUMNS)

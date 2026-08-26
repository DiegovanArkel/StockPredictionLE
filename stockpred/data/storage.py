"""Local storage layer: parquet cache + DuckDB views over the cache.

``cached_fetch`` is the single entry point later tasks should use to pull
data: it calls a provider's fetch function, falls back to the last good
parquet cache on failure (network error or an empty result), and reports
whether the returned frame is stale.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# Logical dataset names that get their own DuckDB view when a matching
# parquet file exists under ``<data_dir>/raw/``.
_VIEW_NAMES = ("prices", "macro", "factors")


def _raw_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "raw"


def _parquet_path(name: str, data_dir: str | Path) -> Path:
    return _raw_dir(data_dir) / f"{name}.parquet"


def save_parquet(df: pd.DataFrame, name: str, data_dir: str | Path) -> Path:
    """Write ``df`` to ``<data_dir>/raw/<name>.parquet``, creating dirs as needed."""
    raw_dir = _raw_dir(data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = _parquet_path(name, data_dir)
    df.to_parquet(path, index=False)
    return path


def load_parquet(name: str, data_dir: str | Path) -> pd.DataFrame | None:
    """Read ``<data_dir>/raw/<name>.parquet``, or ``None`` if it doesn't exist."""
    path = _parquet_path(name, data_dir)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def cached_fetch(
    name: str,
    fetch_fn: Callable[[], pd.DataFrame],
    data_dir: str | Path,
) -> tuple[pd.DataFrame, bool]:
    """Fetch fresh data, falling back to the parquet cache on failure.

    Any exception raised by ``fetch_fn``, or an empty DataFrame it returns,
    counts as a failed fetch: the last cached parquet (if any) is loaded and
    returned with ``stale=True``. A successful, non-empty fetch is saved to
    the cache and returned with ``stale=False``.

    Raises ``RuntimeError`` if the fetch failed and no cache exists.
    """
    df: pd.DataFrame | None
    try:
        df = fetch_fn()
        failed = df is None or df.empty
    except Exception as exc:  # noqa: BLE001 - any provider failure is a cache-fallback trigger
        logger.warning("fetch for %r failed: %s", name, exc)
        df = None
        failed = True

    if not failed:
        save_parquet(df, name, data_dir)
        return df, False

    cached = load_parquet(name, data_dir)
    if cached is None:
        raise RuntimeError(
            f"fetch for {name!r} failed and no cache exists at "
            f"{_parquet_path(name, data_dir)}"
        )
    logger.warning("using stale cache for %r", name)
    return cached, True


def refresh_duckdb(duckdb_path: str | Path, data_dir: str | Path) -> None:
    """(Re)create DuckDB views ``prices``/``macro``/``factors`` over the parquet cache.

    A view is skipped (with a log message, not an error) when its parquet
    file doesn't exist yet.
    """
    duckdb_path = Path(duckdb_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(duckdb_path))
    try:
        for name in _VIEW_NAMES:
            path = _parquet_path(name, data_dir)
            if not path.exists():
                logger.info("skipping view %r: %s not found", name, path)
                continue
            con.execute(
                f"CREATE OR REPLACE VIEW {name} AS "
                f"SELECT * FROM read_parquet('{path.as_posix()}')"
            )
    finally:
        con.close()

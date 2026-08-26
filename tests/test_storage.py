"""Tests for stockpred.data.storage: parquet cache + duckdb views."""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from stockpred.data.storage import (
    cached_fetch,
    load_parquet,
    refresh_duckdb,
    save_parquet,
)


def _sample_prices_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["ASML", "ASML"],
            "close": [650.0, 645.0],
            "adj_close": [648.0, 643.0],
            "volume": [500000, 510000],
        }
    )


# --- parquet round-trip -----------------------------------------------------


def test_save_and_load_parquet_round_trips(tmp_path):
    df = _sample_prices_df()
    save_parquet(df, "prices", tmp_path)
    loaded = load_parquet("prices", tmp_path)
    assert loaded is not None
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), df.reset_index(drop=True))


def test_save_parquet_creates_raw_subdir(tmp_path):
    df = _sample_prices_df()
    save_parquet(df, "prices", tmp_path)
    assert (tmp_path / "raw" / "prices.parquet").exists()


def test_load_parquet_missing_file_returns_none(tmp_path):
    assert load_parquet("does_not_exist", tmp_path) is None


# --- cached_fetch fallback logic --------------------------------------------


def test_cached_fetch_success_saves_cache_and_returns_fresh(tmp_path):
    df = _sample_prices_df()

    def fetch_fn():
        return df

    result, stale = cached_fetch("prices", fetch_fn, tmp_path)
    assert stale is False
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))
    assert (tmp_path / "raw" / "prices.parquet").exists()


def test_cached_fetch_falls_back_to_cache_on_exception(tmp_path):
    df = _sample_prices_df()
    save_parquet(df, "prices", tmp_path)

    def failing_fetch_fn():
        raise ConnectionError("network down")

    result, stale = cached_fetch("prices", failing_fetch_fn, tmp_path)
    assert stale is True
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))


def test_cached_fetch_falls_back_to_cache_on_empty_result(tmp_path):
    df = _sample_prices_df()
    save_parquet(df, "prices", tmp_path)

    def empty_fetch_fn():
        return pd.DataFrame(columns=["date", "ticker", "close", "adj_close", "volume"])

    result, stale = cached_fetch("prices", empty_fetch_fn, tmp_path)
    assert stale is True
    pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))


def test_cached_fetch_raises_when_fetch_fails_and_no_cache(tmp_path):
    def failing_fetch_fn():
        raise ConnectionError("network down")

    with pytest.raises(RuntimeError):
        cached_fetch("prices", failing_fetch_fn, tmp_path)


# --- duckdb views ------------------------------------------------------------


def test_refresh_duckdb_creates_queryable_views(tmp_path):
    prices = _sample_prices_df()
    macro = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "series_id": ["INDPRO"],
            "value": [100.5],
        }
    )
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31"]),
            "mkt_rf": [0.01],
            "smb": [0.001],
            "hml": [-0.002],
            "mom": [0.005],
            "rf": [0.003],
        }
    )
    save_parquet(prices, "prices", tmp_path)
    save_parquet(macro, "macro", tmp_path)
    save_parquet(factors, "factors", tmp_path)

    duckdb_path = tmp_path / "stockpred.duckdb"
    refresh_duckdb(duckdb_path, tmp_path)

    con = duckdb.connect(str(duckdb_path))
    try:
        assert con.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 2
        assert con.execute("SELECT COUNT(*) FROM macro").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM factors").fetchone()[0] == 1
    finally:
        con.close()


def test_refresh_duckdb_skips_missing_parquet(tmp_path):
    prices = _sample_prices_df()
    save_parquet(prices, "prices", tmp_path)

    duckdb_path = tmp_path / "stockpred.duckdb"
    # macro.parquet and factors.parquet do not exist; should not raise.
    refresh_duckdb(duckdb_path, tmp_path)

    con = duckdb.connect(str(duckdb_path))
    try:
        assert con.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 2
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        assert "macro" not in tables
        assert "factors" not in tables
    finally:
        con.close()


def test_refresh_duckdb_is_idempotent_replace(tmp_path):
    prices = _sample_prices_df()
    save_parquet(prices, "prices", tmp_path)
    duckdb_path = tmp_path / "stockpred.duckdb"

    refresh_duckdb(duckdb_path, tmp_path)
    refresh_duckdb(duckdb_path, tmp_path)  # should not raise on re-create

    con = duckdb.connect(str(duckdb_path))
    try:
        assert con.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 2
    finally:
        con.close()

"""End-to-end test for stockpred.pipeline.run (Task 10).

Builds a small synthetic dataset (10 tickers x 16 years of business-daily
prices, plus matching monthly macro + Fama-French factor histories),
writes it into a tmp data_dir's parquet cache, and runs the full pipeline
with ``skip_fetch=True`` against a tmp config.yaml. Small LightGBM params
keep this fast (~1-3 minutes) despite the full walk-forward + production
refit + GARCH stack.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from stockpred import pipeline
from stockpred.data.storage import save_parquet

_N_TICKERS = 10
_N_YEARS = 16
_FAST_LGB_PARAMS = {"n_estimators": 20, "num_leaves": 15, "min_child_samples": 10}

_FRED_LOG_DIFF_SERIES = ["INDPRO", "CPIAUCSL", "M2SL"]
_FRED_DIFF_SERIES = ["UNRATE", "FEDFUNDS", "VIXCLS"]


def _make_prices(n_tickers: int = _N_TICKERS, years: int = _N_YEARS, seed: int = 0) -> pd.DataFrame:
    dates = pd.bdate_range(start="2008-01-01", periods=years * 252)
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_tickers):
        ticker = f"T{i:02d}"
        daily_ret = rng.normal(loc=0.0003, scale=0.015, size=len(dates))
        price = 100.0 * np.cumprod(1.0 + daily_ret)
        volume = rng.integers(100_000, 500_000, size=len(dates)).astype(float)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "close": price,
                    "adj_close": price,
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _make_factors(dates: pd.DatetimeIndex, seed: int = 1) -> pd.DataFrame:
    month_ends = pd.date_range(dates.min(), dates.max() + pd.offsets.MonthEnd(1), freq="ME")
    rng = np.random.default_rng(seed)
    n = len(month_ends)
    return pd.DataFrame(
        {
            "date": month_ends,
            "mkt_rf": rng.normal(0.0, 0.03, size=n),
            "smb": rng.normal(0.0, 0.01, size=n),
            "hml": rng.normal(0.0, 0.01, size=n),
            "mom": rng.normal(0.0, 0.01, size=n),
            "rf": np.full(n, 0.001),
        }
    )


def _make_macro(month_ends: pd.DatetimeIndex, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(month_ends)
    series = {
        "INDPRO": 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.01, n)),
        "CPIAUCSL": 100.0 * np.cumprod(1.0 + rng.normal(0.002, 0.005, n)),
        "M2SL": 1000.0 * np.cumprod(1.0 + rng.normal(0.003, 0.005, n)),
        "UNRATE": np.clip(5.0 + np.cumsum(rng.normal(0.0, 0.1, n)), 2.0, 12.0),
        "FEDFUNDS": np.clip(2.0 + np.cumsum(rng.normal(0.0, 0.1, n)), 0.0, 6.0),
        "VIXCLS": np.clip(18.0 + np.cumsum(rng.normal(0.0, 1.0, n)), 10.0, 60.0),
    }
    frames = [
        pd.DataFrame({"date": month_ends, "series_id": series_id, "value": values})
        for series_id, values in series.items()
    ]
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def _synthetic_dataset() -> dict[str, pd.DataFrame]:
    prices = _make_prices()
    factors = _make_factors(prices["date"])
    macro = _make_macro(factors["date"])
    return {"prices": prices, "factors": factors, "macro": macro}


@pytest.fixture
def _pipeline_env(tmp_path, _synthetic_dataset):
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"

    save_parquet(_synthetic_dataset["prices"], "prices", data_dir)
    save_parquet(_synthetic_dataset["factors"], "factors", data_dir)
    save_parquet(_synthetic_dataset["macro"], "macro", data_dir)

    tickers = sorted(_synthetic_dataset["prices"]["ticker"].unique())
    config = {
        "tickers": tickers,
        "fred_series": _FRED_LOG_DIFF_SERIES + _FRED_DIFF_SERIES,
        "ecb_series": [],
        "data_dir": str(data_dir),
        "artifacts_dir": str(artifacts_dir),
        "duckdb_path": str(data_dir / "stockpred.duckdb"),
        "n_folds": 3,
        "test_window_months": 12,
        "embargo_months": 1,
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    return {"config_path": config_path, "artifacts_dir": artifacts_dir}


class TestPipelineRun:
    def test_run_skip_fetch_produces_all_artifacts(self, _pipeline_env):
        summary = pipeline.run(
            config_path=_pipeline_env["config_path"],
            skip_fetch=True,
            lgb_params=_FAST_LGB_PARAMS,
        )

        artifacts_dir = _pipeline_env["artifacts_dir"]
        expected_files = [
            "forecasts.parquet",
            "oos_history.parquet",
            "backtest.json",
            "diagnostics.json",
            "manifest.json",
        ]
        for name in expected_files:
            assert (artifacts_dir / name).exists(), f"missing artifact: {name}"

        forecasts = pd.read_parquet(artifacts_dir / "forecasts.parquet")
        assert len(forecasts) > 0
        assert len(forecasts) <= _N_TICKERS

        expected_cols = {
            "ticker",
            "as_of",
            "q05",
            "q25",
            "q50",
            "q75",
            "q95",
            "raw_q05",
            "raw_q25",
            "raw_q75",
            "raw_q95",
            "ens_q05",
            "ens_q25",
            "ens_q50",
            "ens_q75",
            "ens_q95",
            "y_q05",
            "y_q25",
            "y_q50",
            "y_q75",
            "y_q95",
            "ann_vol",
            "garch_converged",
        }
        assert expected_cols.issubset(set(forecasts.columns))

        with open(artifacts_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["n_forecasts"] == len(forecasts)

        history = pd.read_parquet(artifacts_dir / "oos_history.parquet")
        assert manifest["n_history_rows"] == len(history)

        with open(artifacts_dir / "backtest.json") as f:
            backtest_summary = json.load(f)
        assert "sharpe" in backtest_summary

        with open(artifacts_dir / "diagnostics.json") as f:
            diagnostics = json.load(f)
        assert "wf_metrics" in diagnostics
        assert "offsets" in diagnostics

        expected_stages = {
            "refresh_duckdb",
            "load_data",
            "features",
            "walk_forward",
            "calibration",
            "backtest",
            "garch",
            "production_forecast",
            "assemble_and_write",
        }
        assert expected_stages.issubset(set(summary["timings"].keys()))
        assert "fetch" not in summary["timings"]  # skipped
        assert summary["n_forecasts"] == len(forecasts)
        assert summary["metrics_keys"] == ["folds", "pooled"]

        # Task 10 review fix: every production forecast row must carry the
        # panel date its features actually came from, distinguishable from
        # a stale ticker's forecast.
        assert "pred_date" in forecasts.columns
        assert forecasts["pred_date"].notna().all()


class TestPipelineExcludesStaleTickers:
    """Task 10 review fix: a ticker whose price feed silently stopped
    updating (or was delisted) must not be predicted from months-stale
    features and shipped indistinguishably from a fresh forecast."""

    _STALE_TICKER = "T00"
    _MONTHS_EARLY = 7  # comfortably past the pipeline's 3-month staleness cutoff

    def test_stale_ticker_excluded_from_forecasts_and_named_in_manifest(
        self, tmp_path, _synthetic_dataset
    ):
        prices = _synthetic_dataset["prices"].copy()
        cutoff = prices["date"].max() - pd.DateOffset(months=self._MONTHS_EARLY)
        stale_mask = (prices["ticker"] == self._STALE_TICKER) & (prices["date"] > cutoff)
        prices = prices[~stale_mask].reset_index(drop=True)

        data_dir = tmp_path / "data"
        artifacts_dir = tmp_path / "artifacts"
        save_parquet(prices, "prices", data_dir)
        save_parquet(_synthetic_dataset["factors"], "factors", data_dir)
        save_parquet(_synthetic_dataset["macro"], "macro", data_dir)

        tickers = sorted(prices["ticker"].unique())
        config = {
            "tickers": tickers,
            "fred_series": _FRED_LOG_DIFF_SERIES + _FRED_DIFF_SERIES,
            "ecb_series": [],
            "data_dir": str(data_dir),
            "artifacts_dir": str(artifacts_dir),
            "duckdb_path": str(data_dir / "stockpred.duckdb"),
            "n_folds": 3,
            "test_window_months": 12,
            "embargo_months": 1,
        }
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)

        summary = pipeline.run(
            config_path=config_path, skip_fetch=True, lgb_params=_FAST_LGB_PARAMS
        )

        forecasts = pd.read_parquet(artifacts_dir / "forecasts.parquet")
        assert self._STALE_TICKER not in set(forecasts["ticker"])
        assert len(forecasts) == _N_TICKERS - 1
        assert summary["n_forecasts"] == len(forecasts)

        with open(artifacts_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert self._STALE_TICKER in manifest["failed_tickers"]

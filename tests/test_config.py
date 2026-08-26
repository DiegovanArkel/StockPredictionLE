"""Tests for stockpred.config.load_config."""

from __future__ import annotations

from pathlib import Path

import pytest

from stockpred.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def test_load_config_default_path_returns_config():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(config, Config)


def test_load_config_has_25_aex_tickers():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(config.tickers, list)
    assert len(config.tickers) == 25
    assert len(set(config.tickers)) == 25  # no duplicates
    assert "ASML" in config.tickers
    assert "ADYEN" in config.tickers


def test_load_config_default_quantiles():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.quantiles == [0.05, 0.25, 0.5, 0.75, 0.95]


def test_load_config_paths_are_path_objects():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert isinstance(config.data_dir, Path)
    assert isinstance(config.artifacts_dir, Path)
    assert isinstance(config.duckdb_path, Path)


def test_load_config_is_frozen():
    config = load_config(DEFAULT_CONFIG_PATH)
    with pytest.raises(Exception):
        config.n_pca_factors = 99  # type: ignore[misc]


def test_load_config_default_values_applied():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.yf_suffix == ".AS"
    assert config.n_pca_factors == 5
    assert config.horizon_months == 1
    assert config.decay_half_life_years == 5.0
    assert config.n_folds == 8
    assert config.embargo_months == 1
    assert config.test_window_months == 12
    assert config.cost_fixed_eur == 2.0
    assert config.cost_bps == 2.61
    assert config.signal_threshold == 0.01
    assert config.loss_tolerance == -0.10


def test_load_config_fred_and_ecb_series_present():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert "INDPRO" in config.fred_series
    assert "VIXCLS" in config.fred_series
    assert "ICP.M.U2.N.000000.4.ANR" in config.ecb_series


def test_load_config_unknown_top_level_key_raises(tmp_path):
    bad_yaml = tmp_path / "bad_config.yaml"
    bad_yaml.write_text(
        """
tickers: [ASML]
fred_series: [INDPRO]
ecb_series: []
data_dir: data
artifacts_dir: artifacts
duckdb_path: data/stockpred.duckdb
totally_unknown_key: 123
"""
    )
    with pytest.raises(ValueError):
        load_config(bad_yaml)


def test_load_config_missing_optional_keys_use_defaults(tmp_path):
    minimal_yaml = tmp_path / "minimal_config.yaml"
    minimal_yaml.write_text(
        """
tickers: [ASML, ADYEN]
fred_series: [INDPRO]
ecb_series: []
data_dir: data
artifacts_dir: artifacts
duckdb_path: data/stockpred.duckdb
"""
    )
    config = load_config(minimal_yaml)
    assert config.yf_suffix == ".AS"
    assert config.quantiles == [0.05, 0.25, 0.5, 0.75, 0.95]
    assert config.n_pca_factors == 5
    assert config.price_start == "2000-01-01"


def test_load_config_missing_required_key_raises(tmp_path):
    incomplete_yaml = tmp_path / "incomplete_config.yaml"
    incomplete_yaml.write_text(
        """
tickers: [ASML]
fred_series: [INDPRO]
ecb_series: []
"""
    )
    with pytest.raises(ValueError):
        load_config(incomplete_yaml)


def test_load_config_string_path_argument_default(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    config = load_config("config.yaml")
    assert isinstance(config, Config)
    assert len(config.tickers) == 25

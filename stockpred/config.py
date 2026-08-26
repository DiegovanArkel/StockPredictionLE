"""Project configuration: a frozen dataclass loaded from ``config.yaml``.

All returns/rates elsewhere in the project are stored as decimal fractions
(0.05 == 5%), never percent.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

# Keys that must be present in the YAML config (no documented default).
_REQUIRED_KEYS = frozenset(
    {
        "tickers",
        "fred_series",
        "ecb_series",
        "data_dir",
        "artifacts_dir",
        "duckdb_path",
    }
)

# YAML keys that map onto Path-typed Config attributes.
_PATH_KEYS = frozenset({"data_dir", "artifacts_dir", "duckdb_path"})

# The ONLY supported quantile family. This is not a stylistic preference:
# the q05/q25/q50/q75/q95 names are hardcoded downstream in the conformal
# layer (stockpred/models/conformal.py operates on the (q05,q95) and
# (q25,q75) pairs by name), the GARCH layer, artifact assembly, and the
# dashboard's quantile tables and fan charts. Configuring a different set
# would produce columns nothing downstream reads and silently break those
# consumers, so it is rejected at load time instead.
_SUPPORTED_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


@dataclass(frozen=True)
class Config:
    """Immutable project configuration.

    Fields with no default must be supplied by ``config.yaml``; fields with a
    default are optional in ``config.yaml`` and fall back to the documented
    value shown here when omitted.
    """

    tickers: list[str]
    fred_series: list[str]
    ecb_series: list[str]
    data_dir: Path
    artifacts_dir: Path
    duckdb_path: Path

    yf_suffix: str = ".AS"
    price_start: str = "2000-01-01"
    n_pca_factors: int = 5
    quantiles: list[float] = field(
        default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95]
    )
    horizon_months: int = 1
    decay_half_life_years: float = 5.0
    n_folds: int = 8
    embargo_months: int = 1
    test_window_months: int = 12
    cost_fixed_eur: float = 2.0
    cost_bps: float = 2.61
    signal_threshold: float = 0.01
    loss_tolerance: float = -0.10


_VALID_KEYS = frozenset(f.name for f in fields(Config))


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and validate a :class:`Config` from a YAML file.

    Unknown top-level keys raise ``ValueError``. Missing required keys (those
    without a documented default) also raise ``ValueError``. Missing optional
    keys fall back to the documented default. Path-typed values are converted
    from strings to :class:`pathlib.Path`.
    """
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a top-level mapping")

    unknown_keys = set(raw) - _VALID_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unknown config key(s) in {path}: {sorted(unknown_keys)}"
        )

    missing_keys = _REQUIRED_KEYS - set(raw)
    if missing_keys:
        raise ValueError(
            f"Missing required config key(s) in {path}: {sorted(missing_keys)}"
        )

    kwargs = dict(raw)
    for key in _PATH_KEYS:
        kwargs[key] = Path(kwargs[key])

    if "quantiles" in kwargs:
        quantiles = kwargs["quantiles"]
        if not isinstance(quantiles, list) or [float(q) for q in quantiles] != _SUPPORTED_QUANTILES:
            raise ValueError(
                f"Unsupported 'quantiles' in {path}: {quantiles!r}. The only "
                f"supported quantile family is {_SUPPORTED_QUANTILES} -- the "
                "q05/q25/q50/q75/q95 names are hardcoded in the conformal "
                "calibration layer, the GARCH layer, artifact assembly and "
                "the dashboard. Omit the key to use the default."
            )

    return Config(**kwargs)

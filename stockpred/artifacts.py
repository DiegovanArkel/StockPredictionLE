"""Forecast assembly + artifact read/write layer (Task 8).

``assemble_forecasts`` combines a production LightGBM quantile forecast
frame (already CQR-calibrated by :func:`stockpred.models.conformal.apply_cqr`)
with a GARCH volatility/return-quantile frame
(:func:`stockpred.models.garch.garch_all`) into one per-ticker forecast
frame: calibrated monthly quantiles, a 50/50 LightGBM/GARCH ensemble, and a
GARCH-derived, ensemble-recentered yearly band.

``write_artifacts``/``load_artifacts`` are the storage boundary between the
modeling pipeline (Tasks 5-9, which may import lightgbm/arch/sklearn) and the
Streamlit dashboard (Task 11), which must not need those heavy packages to
render. Keep this module's *top-level* imports limited to
pandas/json/pathlib/datetime (plus numpy, a lightweight transitive pandas
dependency) so ``load_artifacts`` stays cheap to import; package-version
metadata is looked up lazily inside ``write_artifacts``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_QUANTILE_SUFFIXES = ["q05", "q25", "q50", "q75", "q95"]
_MONTHLY_FAMILY = [f"q{s[1:]}" for s in _QUANTILE_SUFFIXES]  # q05..q95
_ENS_FAMILY = [f"ens_{s}" for s in _QUANTILE_SUFFIXES]
_YEARLY_FAMILY = [f"y_{s}" for s in _QUANTILE_SUFFIXES]

_VERSIONED_PACKAGES = ["pandas", "lightgbm", "arch", "stockpred"]


def _sort_family(df: pd.DataFrame, cols: list[str]) -> None:
    """Enforce non-decreasing order across ``cols`` within each row, in
    place. NaN-only rows (e.g. a ticker missing GARCH data) sort to
    themselves -- ``np.sort`` places NaNs last, which is a no-op when every
    value in the row is NaN.
    """
    arr = df[cols].to_numpy(dtype=float)
    arr = np.sort(arr, axis=1)
    for i, col in enumerate(cols):
        df[col] = arr[:, i]


def assemble_forecasts(
    final_pred: pd.DataFrame,
    garch_df: pd.DataFrame,
    offsets: dict,
    as_of: Any,
) -> pd.DataFrame:
    """Assemble one production forecast row per ticker.

    ``final_pred``: ``[ticker, q05, q25, q50, q75, q95, q05_cal, q25_cal,
    q75_cal, q95_cal]`` -- CQR-calibrated LightGBM monthly quantiles (see
    :func:`stockpred.models.conformal.apply_cqr`; only the tails are
    calibrated, so there is no ``q50_cal``).

    ``garch_df``: :func:`stockpred.models.garch.garch_all` output --
    ``[ticker, ann_vol, m_q05..m_q95, y_q05..y_q95, converged]``.

    ``offsets``: the CQR offsets dict from
    :func:`stockpred.models.conformal.calibrate_from_wf`. Not consumed by
    the math here (``final_pred`` already carries calibrated values) --
    accepted for interface symmetry with ``write_artifacts``/diagnostics,
    which persist it alongside the forecasts it produced.

    ``as_of``: anything ``pandas.Timestamp`` can parse (date, datetime,
    ISO string); stamped onto every row as an ``as_of`` ISO date string
    (``YYYY-MM-DD``).

    Tickers are combined via an outer join on ``ticker``. A ticker absent
    from ``garch_df`` gets NaN GARCH-derived columns, ``garch_converged =
    False``, NaN yearly quantiles, and an ensemble that falls back to the
    pure (calibrated) LightGBM quantiles.

    Returns one row per ticker with columns:
    ``ticker, as_of, q05..q95`` (calibrated LightGBM -- the monthly display
    quantiles), ``raw_q05, raw_q25, raw_q75, raw_q95`` (pre-calibration,
    for diagnostics), ``ens_q05..ens_q95`` (elementwise mean of calibrated
    LightGBM and GARCH month quantiles, or the LightGBM value alone),
    ``y_q05..y_q95`` (GARCH year quantiles re-centered so their median
    equals ``12 * ens_q50``), ``ann_vol``, ``garch_converged``.

    Each of the three quantile families (``q*``, ``ens_q*``, ``y_q*``) is
    sorted to be non-decreasing within a row -- guarding against crossing
    that could in principle arise from averaging two independently
    estimated quantile curves.
    """
    merged = final_pred.merge(garch_df, on="ticker", how="outer", suffixes=("", "_garch"))

    as_of_str = pd.Timestamp(as_of).strftime("%Y-%m-%d")

    out = pd.DataFrame({"ticker": merged["ticker"], "as_of": as_of_str})

    # Monthly display quantiles: calibrated tails, uncalibrated median.
    out["q05"] = merged["q05_cal"]
    out["q25"] = merged["q25_cal"]
    out["q50"] = merged["q50"]
    out["q75"] = merged["q75_cal"]
    out["q95"] = merged["q95_cal"]

    # Pre-calibration LightGBM quantiles, kept for diagnostics.
    out["raw_q05"] = merged["q05"]
    out["raw_q25"] = merged["q25"]
    out["raw_q75"] = merged["q75"]
    out["raw_q95"] = merged["q95"]

    def _ensemble(cal_col: str, garch_col: str) -> np.ndarray:
        cal = out[cal_col].to_numpy(dtype=float) if cal_col in out.columns else merged[cal_col].to_numpy(dtype=float)
        if garch_col in merged.columns:
            g = merged[garch_col].to_numpy(dtype=float)
            return np.where(np.isnan(g), cal, (cal + g) / 2.0)
        return cal

    out["ens_q05"] = _ensemble("q05", "m_q05")
    out["ens_q25"] = _ensemble("q25", "m_q25")
    out["ens_q50"] = _ensemble("q50", "m_q50")
    out["ens_q75"] = _ensemble("q75", "m_q75")
    out["ens_q95"] = _ensemble("q95", "m_q95")

    # Yearly bands: GARCH year quantiles, re-centered so the median equals
    # 12 * ens_q50. Missing GARCH (NaN inputs) propagate to NaN outputs
    # automatically -- no special-casing needed.
    target_median = 12.0 * out["ens_q50"].to_numpy(dtype=float)
    y_q50_raw = merged["y_q50"].to_numpy(dtype=float) if "y_q50" in merged.columns else np.full(len(merged), np.nan)
    recenter = target_median - y_q50_raw
    for suffix in _QUANTILE_SUFFIXES:
        col = f"y_{suffix}"
        if col in merged.columns:
            out[col] = merged[col].to_numpy(dtype=float) + recenter
        else:
            out[col] = np.nan

    out["ann_vol"] = merged["ann_vol"] if "ann_vol" in merged.columns else np.nan

    if "converged" in merged.columns:
        out["garch_converged"] = merged["converged"].fillna(False).astype(bool)
    else:
        out["garch_converged"] = False

    _sort_family(out, _MONTHLY_FAMILY)
    _sort_family(out, _ENS_FAMILY)
    _sort_family(out, _YEARLY_FAMILY)

    return out.sort_values("ticker").reset_index(drop=True)


def _json_default(obj: Any) -> Any:
    """``json.dump(..., default=...)`` handler: converts numpy scalar/array
    types and pandas Timestamps to native JSON-serializable Python types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, default=_json_default, indent=2)


def write_artifacts(
    artifacts_dir: Any,
    forecasts: pd.DataFrame,
    wf_metrics: dict,
    offsets: dict,
    backtest_summary: dict,
    history: pd.DataFrame,
    failures: list,
    staleness: dict,
) -> None:
    """Write the dashboard's entire input surface to ``artifacts_dir``.

    Writes ``forecasts.parquet`` (``forecasts`` as-is), ``oos_history.parquet``
    (``history`` as-is -- calibrated OOS predictions + y_true, for the
    diagnostics page), ``backtest.json`` (``backtest_summary`` as-is),
    ``diagnostics.json`` (``{"wf_metrics": wf_metrics, "offsets": offsets}``),
    and ``manifest.json`` (generation timestamp, row counts, failed tickers,
    stale sources, and package versions).

    Creates ``artifacts_dir`` (and parents) if it doesn't exist. All JSON
    values pass through a ``default=`` handler that converts numpy scalar/
    array types to native Python so the files round-trip through
    ``json.load``.
    """
    import importlib.metadata as importlib_metadata

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    forecasts.to_parquet(artifacts_dir / "forecasts.parquet", index=False)
    history.to_parquet(artifacts_dir / "oos_history.parquet", index=False)

    _write_json(artifacts_dir / "backtest.json", backtest_summary)
    _write_json(artifacts_dir / "diagnostics.json", {"wf_metrics": wf_metrics, "offsets": offsets})

    def _version(pkg: str) -> str | None:
        try:
            return importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            return None

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_forecasts": int(len(forecasts)),
        "n_history_rows": int(len(history)),
        "failed_tickers": list(failures),
        "stale_sources": dict(staleness),
        "versions": {pkg: _version(pkg) for pkg in _VERSIONED_PACKAGES},
    }
    _write_json(artifacts_dir / "manifest.json", manifest)


def load_artifacts(artifacts_dir: Any) -> dict:
    """Read back everything ``write_artifacts`` wrote.

    Returns a dict with keys ``forecasts, oos_history, diagnostics,
    backtest, manifest``. A missing file (including a missing
    ``artifacts_dir``) maps its key to ``None`` -- callers (the dashboard)
    are expected to render a placeholder rather than treat that as an
    error. A *present but corrupt* file raises (a bad parquet footer or
    malformed JSON is a real problem, not a "not generated yet" state).
    """
    artifacts_dir = Path(artifacts_dir)

    def _read_parquet(name: str) -> pd.DataFrame | None:
        path = artifacts_dir / name
        return pd.read_parquet(path) if path.exists() else None

    def _read_json(name: str) -> Any:
        path = artifacts_dir / name
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    return {
        "forecasts": _read_parquet("forecasts.parquet"),
        "oos_history": _read_parquet("oos_history.parquet"),
        "diagnostics": _read_json("diagnostics.json"),
        "backtest": _read_json("backtest.json"),
        "manifest": _read_json("manifest.json"),
    }

"""Macro stationarity transforms + leakage-safe PCA factor extraction.

``build_macro_wide`` turns the tidy long macro frame (see
``stockpred.data.macro``) into a stationary, wide monthly panel.
``MacroFactorExtractor`` then reduces that panel to a handful of PCA
factors -- it MUST be fit only on training-fold months (walk-forward CV in
Task 5): ``.fit`` stores the standardization stats and PCA components,
``.transform`` reuses them on any other slice, so no information from
validation/test months leaks into the factors.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Strictly-positive level series: stationarity via 12-month log-difference.
# Everything else (rates, spreads, indices) uses a simple 1-month difference.
_LOG_DIFF_12M_SERIES = frozenset({"INDPRO", "CPIAUCSL", "M2SL"})

_FFILL_LIMIT = 2
_MAX_MISSING_FRAC = 0.30


def build_macro_wide(macro: pd.DataFrame) -> pd.DataFrame:
    """Pivot tidy macro data to a stationary, wide monthly panel.

    Steps: pivot ``[date, series_id, value]`` to wide, resample to month-end
    (``.last()``), transform each column to stationarity (12m log-diff for
    ``INDPRO``/``CPIAUCSL``/``M2SL``, 1m diff for everything else),
    forward-fill gaps of at most 2 months, then drop columns that are still
    >30% missing.
    """
    wide = macro.pivot(index="date", columns="series_id", values="value").sort_index()
    wide = wide.resample("ME").last()

    transformed = pd.DataFrame(index=wide.index)
    for col in wide.columns:
        if col in _LOG_DIFF_12M_SERIES:
            transformed[col] = np.log(wide[col]).diff(12)
        else:
            transformed[col] = wide[col].diff(1)

    transformed = transformed.ffill(limit=_FFILL_LIMIT)

    missing_frac = transformed.isna().mean()
    keep_cols = missing_frac[missing_frac <= _MAX_MISSING_FRAC].index
    return transformed[keep_cols]


class MacroFactorExtractor:
    """PCA dimensionality reduction over a macro wide panel, with a strict
    fit/transform split so it can be fit per walk-forward fold on train
    months only (see the Task 5 model, which is the consumer).

    ``.fit`` stores per-column mean/std (population std, matching
    ``sklearn.preprocessing.StandardScaler``) and a fitted
    ``sklearn.decomposition.PCA``, using only rows with no missing values.
    ``.transform`` standardizes any other frame with those STORED stats and
    projects it onto the fitted components, returning columns ``F1..Fn``
    (rows with NaN inputs project to NaN, not silently dropped).
    """

    def __init__(self, n_factors: int = 5):
        self.n_factors = n_factors
        self._mean: pd.Series | None = None
        self._std: pd.Series | None = None
        self._pca: PCA | None = None
        self._columns: list[str] | None = None

    @property
    def _fitted(self) -> bool:
        return self._pca is not None

    def fit(self, df: pd.DataFrame) -> "MacroFactorExtractor":
        """Fit standardization stats and PCA on the complete-case rows of
        `df` only. `df` should be train-fold months only -- never the full
        sample -- to avoid leaking validation/test information into the
        factors."""
        clean = df.dropna(how="any")
        if clean.empty:
            raise ValueError("MacroFactorExtractor.fit: no complete (no-NaN) rows to fit on")

        self._columns = list(df.columns)

        n_features = len(self._columns)
        n_rows = len(clean)
        # sklearn's PCA requires n_components <= min(n_samples, n_features);
        # violating that raises an opaque ValueError deep inside sklearn, so
        # cap against both here and raise our own clear error if there's
        # nothing left to fit (e.g. a train fold with fewer complete rows
        # than requested factors).
        effective_n_factors = min(self.n_factors, n_features, n_rows)
        if effective_n_factors < 1:
            raise ValueError(
                f"MacroFactorExtractor.fit: cannot fit any PCA components -- "
                f"only {n_rows} complete row(s) and {n_features} feature(s) "
                "after dropping NaNs"
            )
        if effective_n_factors < self.n_factors:
            warnings.warn(
                f"n_factors={self.n_factors} exceeds available rank "
                f"(n_features={n_features}, complete_rows={n_rows}); "
                f"capping to {effective_n_factors}",
                UserWarning,
                stacklevel=2,
            )

        self._mean = clean.mean(axis=0)
        std = clean.std(axis=0, ddof=0)
        self._std = std.where(std != 0, 1.0)  # guard constant columns

        standardized = (clean - self._mean) / self._std
        self._pca = PCA(n_components=effective_n_factors, random_state=0)
        self._pca.fit(standardized.to_numpy())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize `df` with the STORED fit stats and project onto the
        fitted PCA components. Rows with any NaN feature produce NaN
        factors (not dropped), so the output index always matches `df`."""
        if not self._fitted:
            raise RuntimeError("MacroFactorExtractor.transform called before fit")

        assert self._mean is not None and self._std is not None and self._columns is not None
        aligned = df[self._columns]
        standardized = (aligned - self._mean) / self._std

        components = self._pca.components_  # (n_factors, n_features)
        pca_mean = self._pca.mean_
        projected = (standardized.to_numpy() - pca_mean) @ components.T

        n_factors = components.shape[0]
        return pd.DataFrame(
            projected,
            index=df.index,
            columns=[f"F{i + 1}" for i in range(n_factors)],
        )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: `fit(df)` then `transform(df)`. Only appropriate when
        `df` is itself already a leakage-safe (train-only) slice."""
        self.fit(df)
        return self.transform(df)

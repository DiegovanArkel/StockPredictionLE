"""Tests for stockpred.features.macro_pca: macro stationarity transforms
(``build_macro_wide``) and the leakage-safe PCA extractor
(``MacroFactorExtractor``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.features.macro_pca import MacroFactorExtractor, build_macro_wide


def _month_ends(n: int, start: str = "2018-01-31") -> pd.DatetimeIndex:
    first = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
    return pd.DatetimeIndex([first + pd.offsets.MonthEnd(i) for i in range(n)])


def _tidy_macro(series: dict[str, list[float]], dates: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    for series_id, values in series.items():
        assert len(values) == len(dates)
        frames.append(pd.DataFrame({"date": dates, "series_id": series_id, "value": values}))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# build_macro_wide: pivot + resample
# ---------------------------------------------------------------------------


def test_build_macro_wide_pivots_to_wide_monthly_index():
    dates = _month_ends(6)
    macro = _tidy_macro(
        {"FEDFUNDS": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5], "UNRATE": [4.0, 4.0, 3.9, 3.9, 3.8, 3.8]},
        dates,
    )

    wide = build_macro_wide(macro)

    assert isinstance(wide.index, pd.DatetimeIndex)
    assert list(wide.index) == list(dates)
    # both series survive as columns (transformed, not raw)
    assert {"FEDFUNDS", "UNRATE"}.issubset(set(wide.columns))


# ---------------------------------------------------------------------------
# Stationarity transforms
# ---------------------------------------------------------------------------


def test_indpro_uses_12m_log_diff():
    # n large enough that the 12 leading NaNs from the 12m diff stay under
    # the 30%-missing column-drop threshold (12/45 ~= 27%).
    n = 45
    dates = _month_ends(n)
    # constant monthly growth rate -> exact, predictable 12m log-diff
    growth = 1.01
    levels = [100.0 * (growth**i) for i in range(n)]
    macro = _tidy_macro({"INDPRO": levels}, dates)

    wide = build_macro_wide(macro)

    expected = np.log(np.array(levels[12:])) - np.log(np.array(levels[:-12]))
    got = wide["INDPRO"].iloc[12:].to_numpy()
    assert got == pytest.approx(expected, abs=1e-9)
    # first 12 months undefined (no 12m-ago observation)
    assert wide["INDPRO"].iloc[:12].isna().all()


def test_rate_series_uses_1m_diff():
    n = 6
    dates = _month_ends(n)
    values = [1.0, 1.25, 1.10, 1.10, 1.30, 1.35]
    macro = _tidy_macro({"FEDFUNDS": values}, dates)

    wide = build_macro_wide(macro)

    expected = pd.Series(values).diff(1).to_numpy()
    got = wide["FEDFUNDS"].to_numpy()
    np.testing.assert_allclose(got, expected, equal_nan=True)


def test_cpiaucsl_and_m2sl_also_use_12m_log_diff():
    n = 45
    dates = _month_ends(n)
    levels_cpi = [100.0 * (1.002**i) for i in range(n)]
    levels_m2 = [1000.0 * (1.005**i) for i in range(n)]
    macro = _tidy_macro({"CPIAUCSL": levels_cpi, "M2SL": levels_m2}, dates)

    wide = build_macro_wide(macro)

    expected_cpi = np.log(np.array(levels_cpi[12:])) - np.log(np.array(levels_cpi[:-12]))
    expected_m2 = np.log(np.array(levels_m2[12:])) - np.log(np.array(levels_m2[:-12]))
    assert wide["CPIAUCSL"].iloc[12:].to_numpy() == pytest.approx(expected_cpi, abs=1e-9)
    assert wide["M2SL"].iloc[12:].to_numpy() == pytest.approx(expected_m2, abs=1e-9)


# ---------------------------------------------------------------------------
# ffill(limit=2) and >30% missing column drop
# ---------------------------------------------------------------------------


def test_ffill_limit_2_fills_short_gaps_not_long_gaps():
    n = 10
    dates = _month_ends(n)
    # FEDFUNDS with a 2-month gap (fillable) and VIXCLS with a 3-month gap (not fillable)
    fedfunds = [1.0, 1.1, 1.2, np.nan, np.nan, 1.5, 1.6, 1.7, 1.8, 1.9]
    vix = [15.0, 16.0, 17.0, np.nan, np.nan, np.nan, 20.0, 21.0, 22.0, 23.0]
    macro = _tidy_macro({"FEDFUNDS": fedfunds, "VIXCLS": vix}, dates)

    wide = build_macro_wide(macro)

    ff = wide["FEDFUNDS"]
    # the 2-month raw gap becomes a 3-long NaN run in the 1m-diff series;
    # ffill(limit=2) fills the first 2 (from the diff=0.1 before the gap),
    # the 3rd stays NaN.
    assert ff.iloc[3] == pytest.approx(0.1)
    assert ff.iloc[4] == pytest.approx(0.1)
    assert pd.isna(ff.iloc[5])

    vx = wide["VIXCLS"]
    # the 3-month raw gap becomes a 4-long NaN run in the 1m-diff series;
    # ffill(limit=2) fills only the first 2, 2 remain NaN.
    assert vx.iloc[3] == pytest.approx(1.0)
    assert vx.iloc[4] == pytest.approx(1.0)
    assert pd.isna(vx.iloc[5])
    assert pd.isna(vx.iloc[6])


def test_columns_over_30pct_missing_after_transform_are_dropped():
    n = 20
    dates = _month_ends(n)
    good = [1.0 + 0.01 * i for i in range(n)]
    # mostly-missing series -- even after ffill(limit=2), > 30% NaN remains
    sparse = [1.0] + [np.nan] * (n - 1)
    macro = _tidy_macro({"FEDFUNDS": good, "SPARSE_SERIES": sparse}, dates)

    wide = build_macro_wide(macro)

    assert "FEDFUNDS" in wide.columns
    assert "SPARSE_SERIES" not in wide.columns


# ---------------------------------------------------------------------------
# MacroFactorExtractor
# ---------------------------------------------------------------------------


def _rank3_synthetic_wide(n: int, n_cols: int = 8, seed: int = 0) -> pd.DataFrame:
    """A wide macro-like frame whose columns are near-exact linear
    combinations of 3 latent factors (rank-3 structure) plus tiny noise, so
    PCA(n_components>=3) should explain >90% of variance."""
    rng = np.random.default_rng(seed)
    dates = _month_ends(n)
    latents = rng.normal(size=(n, 3))
    loadings = rng.normal(size=(3, n_cols))
    noise = rng.normal(scale=0.01, size=(n, n_cols))
    data = latents @ loadings + noise
    return pd.DataFrame(data, index=dates, columns=[f"col{i}" for i in range(n_cols)])


def test_pca_explains_over_90pct_variance_on_rank3_synthetic():
    wide = _rank3_synthetic_wide(120)

    extractor = MacroFactorExtractor(n_factors=5)
    extractor.fit(wide)

    explained = extractor._pca.explained_variance_ratio_
    assert explained[:3].sum() > 0.90


def test_fit_transform_returns_f1_to_fn_indexed_by_date():
    wide = _rank3_synthetic_wide(60)
    extractor = MacroFactorExtractor(n_factors=4)

    factors = extractor.fit_transform(wide)

    assert list(factors.columns) == ["F1", "F2", "F3", "F4"]
    assert list(factors.index) == list(wide.index)


def test_transform_on_unseen_dates_works():
    wide = _rank3_synthetic_wide(100)
    train = wide.iloc[:70]
    unseen = wide.iloc[70:]

    extractor = MacroFactorExtractor(n_factors=3)
    extractor.fit(train)
    factors = extractor.transform(unseen)

    assert list(factors.index) == list(unseen.index)
    assert factors.shape == (len(unseen), 3)
    assert factors.notna().all().all()


def test_transform_before_fit_raises_runtime_error():
    wide = _rank3_synthetic_wide(20)
    extractor = MacroFactorExtractor(n_factors=3)

    with pytest.raises(RuntimeError):
        extractor.transform(wide)


def test_n_factors_capped_when_exceeding_n_features():
    wide = _rank3_synthetic_wide(30, n_cols=3)
    extractor = MacroFactorExtractor(n_factors=5)

    with pytest.warns(UserWarning):
        extractor.fit(wide)

    factors = extractor.transform(wide)
    assert list(factors.columns) == ["F1", "F2", "F3"]


def test_n_factors_capped_when_fewer_complete_rows_than_n_factors():
    """A train fold can have more feature columns than complete (no-NaN)
    rows -- sklearn's PCA raises an opaque ValueError if n_components
    exceeds min(n_samples, n_features), so this must be capped (with a
    warning) the same way an n_features shortfall is."""
    wide = _rank3_synthetic_wide(10, n_cols=8)
    sparse = wide.copy()
    sparse.iloc[2:, 0] = np.nan  # only rows 0,1 remain complete-case

    extractor = MacroFactorExtractor(n_factors=5)
    with pytest.warns(UserWarning):
        extractor.fit(sparse)

    factors = extractor.transform(wide)
    assert list(factors.columns) == ["F1", "F2"]


def test_fit_raises_clear_error_when_no_rows_survive_dropna():
    wide = _rank3_synthetic_wide(5, n_cols=4)
    all_nan_one_col = wide.copy()
    all_nan_one_col.iloc[:, 0] = np.nan  # every row now has a NaN

    extractor = MacroFactorExtractor(n_factors=3)
    with pytest.raises(ValueError, match="complete"):
        extractor.fit(all_nan_one_col)


def test_fit_drops_nan_rows_before_fitting():
    wide = _rank3_synthetic_wide(50)
    wide_with_nan = wide.copy()
    wide_with_nan.iloc[0, 0] = np.nan
    wide_with_nan.iloc[5, 2] = np.nan

    extractor = MacroFactorExtractor(n_factors=3)
    # must not raise despite NaN rows present
    extractor.fit(wide_with_nan)
    factors = extractor.transform(wide)
    assert factors.notna().all().all()


def test_transform_uses_stored_fit_stats_not_transform_input_stats():
    """Leakage guard: transform on a slice with a very different mean/std
    than the fit slice must still use the FIT slice's stored mean/std."""
    wide = _rank3_synthetic_wide(60)
    train = wide.iloc[:40]

    extractor = MacroFactorExtractor(n_factors=3)
    extractor.fit(train)

    shifted = wide.iloc[40:] + 1000.0  # wildly different distribution
    manual_standardized = (shifted - extractor._mean) / extractor._std
    expected = (
        manual_standardized.to_numpy() - extractor._pca.mean_
    ) @ extractor._pca.components_.T

    got = extractor.transform(shifted).to_numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-8)

    # and it must NOT equal what you'd get standardizing with the shifted
    # slice's OWN mean/std (that would be leakage from transform data)
    own_mean = shifted.mean(axis=0)
    own_std = shifted.std(axis=0, ddof=0)
    leaky_standardized = (shifted - own_mean) / own_std
    leaky = (leaky_standardized.to_numpy() - extractor._pca.mean_) @ extractor._pca.components_.T
    assert not np.allclose(got, leaky)

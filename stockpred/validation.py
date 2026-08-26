"""Validation utilities: purged walk-forward CV, recency-decay sample
weights, and the deflated Sharpe ratio.

All returns are decimal fractions (0.05 == 5%), never percent.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from scipy import stats

# Euler-Mascheroni constant, used in the expected-max-Sharpe approximation.
_EULER_MASCHERONI = 0.5772156649015329
_PSR_EPSILON = 1e-12


def purged_walk_forward(
    dates: pd.Series,
    n_folds: int,
    test_window_months: int,
    embargo_months: int,
    min_train_months: int = 120,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Purged, embargoed walk-forward CV splits over a monthly panel.

    ``dates`` is the panel's ``date`` column: a month-end Timestamp repeated
    once per ticker for every ticker present that month (a real long panel,
    not a deduplicated date index). Splits are computed over the *unique*
    sorted months and then expanded back out to every row that shares each
    selected month, so every fold's train/test index arrays cover all
    tickers of the months they select.

    Fold construction: walking backward from the most recent month, carve
    out ``n_folds`` consecutive, non-overlapping test blocks of
    ``test_window_months`` months each (the most recent block is the last
    ``test_window_months`` months of the sample). If the sample doesn't
    contain enough months for the requested number of blocks, fewer folds
    are produced. Folds are then yielded in chronological order of their
    test block (oldest test block first).

    For each fold, the training set is every unique month ``m`` with::

        m <= test_start - (embargo_months + 1) months

    The ``+ 1`` buys back the one-month forecast horizon: a training
    observation at month ``t`` has its *target* realized during month
    ``t + 1``, so it must be excluded whenever ``t + 1`` would otherwise
    fall inside (or after) the embargo window preceding the test block.
    Folds whose resulting train set has fewer than ``min_train_months``
    unique months are skipped entirely (no fold is yielded for them).

    Returns positional index arrays (``np.ndarray[int]``) into ``dates`` as
    passed in -- i.e. into ``dates.to_numpy()`` order -- regardless of
    ``dates``'s own pandas label index.
    """
    dates = pd.Series(dates).reset_index(drop=True)
    dates_values = dates.to_numpy()
    unique_months = pd.DatetimeIndex(sorted(dates.unique()))
    n_months = len(unique_months)

    # Carve test blocks walking backward from the end of the sample.
    blocks: list[tuple[int, int]] = []
    end = n_months
    for _ in range(n_folds):
        start = end - test_window_months
        if start < 0:
            break
        blocks.append((start, end))
        end = start
    blocks.reverse()  # chronological order: oldest test block first

    for start, end in blocks:
        test_months = unique_months[start:end]
        test_start = test_months[0]
        cutoff = test_start - pd.DateOffset(months=embargo_months + 1)
        train_months = unique_months[unique_months <= cutoff]

        if len(train_months) < min_train_months:
            continue

        train_mask = np.isin(dates_values, train_months.to_numpy())
        test_mask = np.isin(dates_values, test_months.to_numpy())
        train_idx = np.nonzero(train_mask)[0]
        test_idx = np.nonzero(test_mask)[0]
        yield train_idx, test_idx


def decay_weights(dates: pd.Series, half_life_years: float) -> np.ndarray:
    """Exponential recency-decay sample weights.

    Age is measured in years from the maximum date in ``dates``::

        age_years = (max(dates) - date).days / 365.25
        weight = 0.5 ** (age_years / half_life_years)

    The resulting weights are normalized so ``weight.mean() == 1`` (this
    rescaling doesn't change the ratio between any two weights, so a
    date exactly one half-life older than another still gets exactly half
    its weight).
    """
    dates = pd.Series(dates)
    max_date = dates.max()
    age_years = (max_date - dates).dt.days / 365.25
    weights = 0.5 ** (age_years / half_life_years)
    weights = weights.to_numpy(dtype=float)
    return weights / weights.mean()


def deflated_sharpe_ratio(
    returns: np.ndarray, n_trials: int, periods_per_year: int = 12
) -> dict[str, float]:
    """Probabilistic and deflated Sharpe ratio (Bailey & Lopez de Prado 2014).

    ``returns`` are per-period decimal returns (e.g. monthly). Keys in the
    returned dict:

    - ``sharpe``: annualized Sharpe ratio, ``mean/std * sqrt(periods_per_year)``
      with ``std`` using ``ddof=1``.
    - ``psr``: probability the *true* per-period Sharpe ratio exceeds 0,
      accounting for the non-normality of the returns::

          PSR = Phi( SR * sqrt(n - 1) / sqrt(1 - skew*SR + (kurt-1)/4*SR^2) )

      where ``SR`` is the *per-period* (non-annualized) Sharpe ratio, and
      ``skew``/``kurt`` are the sample skewness and *non-excess* kurtosis
      (Gaussian returns => kurt = 3) of ``returns``.
    - ``dsr``: as PSR, but benchmarked against the expected maximum Sharpe
      ratio one would observe by chance across ``n_trials`` independent
      strategies (the "deflation"), instead of against 0::

          SR0 = std_SR * ( (1-gamma)*Phi^-1(1 - 1/n_trials)
                            + gamma*Phi^-1(1 - 1/(n_trials*e)) )
          DSR = Phi( (SR - SR0) * sqrt(n-1) / sqrt(1 - skew*SR + (kurt-1)/4*SR^2) )

      ``gamma`` is the Euler-Mascheroni constant and ``std_SR`` is the
      (Mertens 2002) asymptotic standard error of the per-period Sharpe
      ratio estimator, ``sqrt((1 - skew*SR + (kurt-1)/4*SR^2) / (n-1))`` --
      i.e. the same expression under the PSR square root, rescaled by
      ``n - 1``. With ``n_trials == 1`` there is no multiple-testing
      inflation to correct for, so ``SR0 = 0`` and ``dsr == psr``.

    Edge cases:

    - ``n_trials < 1`` raises ``ValueError``.
    - Fewer than 3 observations, or zero (non-finite) return std, are
      uninformative: returns ``{"sharpe": 0.0, "psr": 0.5, "dsr": 0.5}``.
    - If the term under the PSR/DSR square root is <= 0 (possible with
      extreme sample skew/kurtosis in small samples), it is clamped to a
      small positive epsilon (1e-12) to keep the ratio finite.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    returns = np.asarray(returns, dtype=float)
    n = returns.size

    if n < 3:
        return {"sharpe": 0.0, "psr": 0.5, "dsr": 0.5}

    std = returns.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return {"sharpe": 0.0, "psr": 0.5, "dsr": 0.5}

    mean = returns.mean()
    sr_period = mean / std
    sharpe = sr_period * np.sqrt(periods_per_year)

    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns, fisher=False)  # non-excess: Gaussian -> 3

    term = 1.0 - skew * sr_period + (kurt - 1.0) / 4.0 * sr_period**2
    if term <= 0:
        term = _PSR_EPSILON
    denom = np.sqrt(term)

    psr = stats.norm.cdf(sr_period * np.sqrt(n - 1) / denom)

    if n_trials == 1:
        sr0 = 0.0
    else:
        std_sr = np.sqrt(term / (n - 1))
        sr0 = std_sr * (
            (1 - _EULER_MASCHERONI) * stats.norm.ppf(1 - 1 / n_trials)
            + _EULER_MASCHERONI * stats.norm.ppf(1 - 1 / (n_trials * np.e))
        )

    dsr = stats.norm.cdf((sr_period - sr0) * np.sqrt(n - 1) / denom)

    return {"sharpe": float(sharpe), "psr": float(psr), "dsr": float(dsr)}

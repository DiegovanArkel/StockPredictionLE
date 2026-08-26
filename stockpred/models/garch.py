"""GJR-GARCH(1,1,1) skew-t volatility/return simulation layer.

``fit_and_simulate`` fits a GJR-GARCH(1,1,1) model with a skew-t innovation
distribution (``arch_model(returns * 100, vol="GARCH", p=1, o=1, q=1,
dist="skewt")`` -- arch works in percent space internally, so returns are
scaled up by 100 before fitting and every scaled quantity is divided back
down to decimals before being returned) to a series of daily simple returns,
then simulates forward to produce an annualized volatility forecast plus
compounded-return quantile bands at a 1-month (21 trading day) and 1-year
(252 trading day) horizon.

Reproducibility
----------------
``arch``'s simulation-based forecasting draws innovations from
``model.distribution.simulate(params)``, which is a closure over the
distribution's own ``Generator``/``RandomState`` (see
``arch.univariate.distribution.Distribution.__init__``). The public
``forecast(..., rng=...)`` hook lets you substitute a fully custom sampler,
but reimplementing skew-t sampling ourselves would be redundant --
``arch``'s own distribution object already knows how. Instead we reseed that
generator in place (``res.model.distribution._generator = ...``) immediately
before each ``forecast(method="simulation", ...)`` call. This was verified
against the installed ``arch==8.0.0`` source: ``forecast()`` only replaces
its default rng with ``self._distribution.simulate(dp)`` when ``rng is
None``, and that closure reads from ``self._generator`` on every draw, so
reseeding it before the call makes the whole simulation path deterministic
(confirmed empirically: two forecasts with the generator reset to the same
seed produce bit-identical ``simulations.values``).

Single-simulation reuse for month + year quantiles
---------------------------------------------------
``arch``'s simulation forecast exposes the full per-step simulated paths via
``forecast.simulations.values``, shape ``(1, n_sims, horizon)`` (see
``ARCHModelForecast.simulations``). Rather than simulating separately at
horizon=21 and horizon=252, we simulate once at
``horizon=max(horizon_days, 252)`` and read the month quantiles off the
first ``horizon_days`` columns of that same simulated path, and the year
quantiles off the first 252 columns -- both are prefixes of one shared
sample of daily-return paths, so this is exact (not an approximation) and
saves a full extra simulation pass.
"""

from __future__ import annotations

import logging
import warnings
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

logger = logging.getLogger(__name__)

_QUANTILE_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
_MIN_OBS = 500
_YEAR_HORIZON = 252
_TRADING_DAYS = 252


@dataclass
class GarchResult:
    ann_vol_forecast: float
    month_quantiles: dict[str, float]
    year_quantiles: dict[str, float]
    converged: bool


def _quantile_col(level: float) -> str:
    return f"q{int(round(level * 100)):02d}"


def _quantile_dict(values: np.ndarray) -> dict[str, float]:
    """Quantiles of ``values`` at the project-standard levels, keyed q05..q95."""
    qs = np.quantile(values, _QUANTILE_LEVELS)
    return {_quantile_col(level): float(v) for level, v in zip(_QUANTILE_LEVELS, qs)}


def _compound(paths: np.ndarray) -> np.ndarray:
    """Per-row compounded return: prod(1 + r_t) - 1, over axis=1 (time)."""
    return np.prod(1.0 + paths, axis=1) - 1.0


def _bootstrap_fallback(
    values: np.ndarray, horizon_days: int, n_sims: int, seed: int
) -> GarchResult:
    """Empirical bootstrap of daily returns: seeded resampling with
    replacement, compounded over the horizon, quantiled. Used when the
    daily-return history is too short to fit GARCH on, or when the fit
    fails/doesn't converge.
    """
    rng = np.random.default_rng(seed)
    ann_vol = float(np.std(values, ddof=0) * np.sqrt(_TRADING_DAYS))

    def _sim_quantiles(horizon: int) -> dict[str, float]:
        draws = rng.choice(values, size=(n_sims, horizon), replace=True)
        return _quantile_dict(_compound(draws))

    return GarchResult(
        ann_vol_forecast=ann_vol,
        month_quantiles=_sim_quantiles(horizon_days),
        year_quantiles=_sim_quantiles(_YEAR_HORIZON),
        converged=False,
    )


def fit_and_simulate(
    daily_returns: pd.Series,
    horizon_days: int = 21,
    n_sims: int = 5000,
    seed: int = 0,
) -> GarchResult:
    """Fit a GJR-GARCH(1,1,1) skew-t model to daily simple returns (decimals)
    and simulate forward to get an annualized volatility forecast plus
    month/year compounded-return quantile bands.

    Falls back to an empirical bootstrap (``converged=False``) when there
    are fewer than 500 non-NaN daily observations, or when the GARCH fit
    raises or fails to converge.

    Raises ``ValueError`` if ``daily_returns`` has zero non-NaN observations
    (there is nothing to bootstrap from either).
    """
    clean = pd.Series(daily_returns).dropna().to_numpy(dtype=float)

    if len(clean) == 0:
        raise ValueError("fit_and_simulate: daily_returns has no non-NaN observations")

    if len(clean) < _MIN_OBS:
        return _bootstrap_fallback(clean, horizon_days, n_sims, seed)

    try:
        model = arch_model(clean * 100.0, vol="GARCH", p=1, o=1, q=1, dist="skewt")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp="off")
        if res.convergence_flag != 0:
            raise RuntimeError(
                f"GARCH fit did not converge (convergence_flag={res.convergence_flag})"
            )
    except Exception as exc:  # noqa: BLE001 - any fit failure triggers the fallback
        logger.warning("GARCH fit failed, using bootstrap fallback: %s", exc)
        return _bootstrap_fallback(clean, horizon_days, n_sims, seed)

    sim_horizon = max(horizon_days, _YEAR_HORIZON)
    res.model.distribution._generator = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim_forecast = res.forecast(
            horizon=sim_horizon, method="simulation", simulations=n_sims, reindex=False
        )
    # simulations.values shape is (1, n_sims, sim_horizon); scale back from
    # percent to decimal.
    paths = sim_forecast.simulations.values[0] / 100.0

    month_quantiles = _quantile_dict(_compound(paths[:, :horizon_days]))
    year_quantiles = _quantile_dict(_compound(paths[:, :_YEAR_HORIZON]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        analytic_forecast = res.forecast(horizon=horizon_days, method="analytic", reindex=False)
    # variance forecast is in percent^2 space (row -1 is the forecast made
    # from the last observation); mean over the horizon, scale back to
    # decimal variance, annualize.
    mean_daily_var = float(np.mean(analytic_forecast.variance.values[-1])) / 100.0**2
    ann_vol = float(np.sqrt(mean_daily_var) * np.sqrt(_TRADING_DAYS))

    return GarchResult(
        ann_vol_forecast=ann_vol,
        month_quantiles=month_quantiles,
        year_quantiles=year_quantiles,
        converged=True,
    )


def _ticker_seed(ticker: str) -> int:
    """Stable per-ticker seed so results don't depend on universe order."""
    return zlib.crc32(ticker.encode("utf-8"))


def garch_all(prices: pd.DataFrame, lookback_years: int = 8) -> pd.DataFrame:
    """Run :func:`fit_and_simulate` per ticker on the trailing
    ``lookback_years`` of daily ``adj_close`` returns from a Task-2 tidy
    price frame ``[date, ticker, close, adj_close, volume]``.

    Returns one row per ticker: ``ticker, ann_vol, m_q05..m_q95,
    y_q05..y_q95, converged``. Tickers whose fit raises unexpectedly (beyond
    the bootstrap fallback already handled inside ``fit_and_simulate``) are
    logged and skipped.
    """
    columns = (
        ["ticker", "ann_vol"]
        + [f"m_{_quantile_col(level)}" for level in _QUANTILE_LEVELS]
        + [f"y_{_quantile_col(level)}" for level in _QUANTILE_LEVELS]
        + ["converged"]
    )

    dates = pd.to_datetime(prices["date"])
    cutoff = dates.max() - pd.DateOffset(years=lookback_years)

    rows: list[dict] = []
    for ticker, group in prices.assign(date=dates).sort_values("date").groupby("ticker"):
        windowed = group.loc[group["date"] >= cutoff]
        returns = windowed["adj_close"].pct_change().dropna()

        try:
            result = fit_and_simulate(returns, seed=_ticker_seed(ticker))
        except Exception as exc:  # noqa: BLE001 - keep other tickers processing
            logger.warning("garch_all: skipping ticker %r due to unexpected error: %s", ticker, exc)
            continue

        row = {"ticker": ticker, "ann_vol": result.ann_vol_forecast}
        row.update({f"m_{k}": v for k, v in result.month_quantiles.items()})
        row.update({f"y_{k}": v for k, v in result.year_quantiles.items()})
        row["converged"] = result.converged
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)

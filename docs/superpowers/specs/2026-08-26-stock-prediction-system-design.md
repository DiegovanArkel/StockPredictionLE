# StockPredictionLE — Design

Date: 2026-08-26
Status: Approved (design supplied by user in initial brief; decisions below are defaults chosen within that brief)

## Purpose

A €0/month probabilistic stock-return forecasting system for AEX (Amsterdam) stocks:
monthly-horizon quantile forecasts from a factor-augmented gradient-boosting model,
calibrated by GARCH simulation + conformal prediction, visualized as fan charts in a
Streamlit dashboard that reads precomputed Parquet/JSON artifacts produced by a weekly
GitHub Actions pipeline. Educational / decision-support, not financial advice.

## Components

1. **Data layer** (`stockpred/data/`)
   - Prices: **yfinance primary** (`<TICKER>.AS`). Verified 2026-08-26: pandas-datareader
     0.11.1 has removed its Stooq module and Stooq's CSV endpoints are behind a JS anti-bot
     challenge, so the brief's Stooq-first ordering is inverted; a Stooq adapter is kept as
     best-effort secondary and the cache is the final fallback.
   - Macro: FRED (keyless `fredgraph.csv` endpoint by default; `fredapi` if `FRED_API_KEY` set),
     ECB via `ecbdata`.
   - Factors: Fama-French **Europe** 3 factors + momentum via pandas-datareader `famafrench`.
   - Storage: raw pulls cached as Parquet under `data/raw/`; DuckDB database `data/stockpred.duckdb`
     with clean tables; final model artifacts under `artifacts/` (Parquet + JSON manifest).

2. **Workhorse model** (`stockpred/models/workhorse.py`)
   - PCA: 5 factors extracted from standardized macro panel (Stock-Watson / FAVAR-lite).
   - Stock features: momentum (1/3/6/12m, 12-1), realized volatility (1/3/12m), Amihud illiquidity,
     dollar-volume, plus market beta and FF factor exposures (rolling).
   - LightGBM with quantile loss at q = {0.05, 0.25, 0.50, 0.75, 0.95}, monthly horizon,
     exponential decay sample weighting (half-life ≈ 5 years), 15–25y training window.
   - Validation: walk-forward with purge + embargo; report OOS R² vs historical-mean benchmark.

3. **Calibration layer** (`stockpred/models/garch.py`, `stockpred/models/conformal.py`)
   - GJR-GARCH(1,1) with skew-t errors per stock (`arch`), fit on 5–10y daily returns;
     simulate 1-month-ahead return paths for distribution shape and vol forecasts.
   - Conformalized Quantile Regression (CQR, split-conformal on the walk-forward calibration
     window) wrapped around the LightGBM quantiles. **Decision:** implemented directly
     (~60 lines, transparent, testable) rather than pulling the Nixtla stack in as a dependency.
   - Success criterion: empirical coverage of 90% bands ≈ 90% OOS.

4. **Benchmarks / ensemble** (`stockpred/models/benchmarks.py`)
   - Historical mean and EWMA baselines always computed.
   - Chronos-Bolt zero-shot is **optional** behind `--with-chronos` (torch is heavy);
     only simple-averaged into the ensemble if it beats the workhorse in walk-forward tests.

5. **Backtest** (`stockpred/backtest.py`)
   - Decision rule: long when median forecast > threshold AND q05 > loss tolerance;
     monthly rebalance; costs default to DEGIRO-like €2 + 0.0261% per trade (configurable).
   - **Decision:** implemented in pandas/numpy directly (with Deflated Sharpe per
     Bailey–López de Prado) instead of vectorbt, to avoid its numpy pins on a €0 stack.

6. **Pipeline** (`stockpred/pipeline.py`, CLI `python -m stockpred`)
   - `refresh-data` → `build-features` → `train` → `calibrate` → `forecast` → `backtest` → `write-artifacts`.
   - GitHub Actions weekly cron (`.github/workflows/weekly.yml`) runs the full pipeline and
     commits `artifacts/` back to the repo. (Upload/remote setup left to user per brief.)

7. **Dashboard** (`app/streamlit_app.py`)
   - Reads only `artifacts/` (no live pulls). Per-stock fan charts (plotly) at monthly horizon,
     wide humble yearly scenario bands, model diagnostics page (coverage, OOS R², backtest),
     prominent not-financial-advice caveat.

## Defaults chosen (flagged for user review in final report)

- Universe: current AEX-25 constituents (hardcoded list in `config.yaml`, editable).
- FRED works keyless via fredgraph.csv; `FRED_API_KEY` optional.
- Broker cost default: DEGIRO Basic-like; parameterized in config.
- Python managed with `uv` (venv + lockfile), Python ≥3.11.
- Fama-French Europe factors (not US) since the universe is European.

## Error handling

- Every data pull: retry ×3 with backoff → provider fallback → last cached Parquet with a
  staleness warning recorded in the artifact manifest. Pipeline fails loudly only if no cache exists.
- Tickers that fail to resolve are reported in the manifest, not silently dropped.

## Testing

- pytest unit tests for feature construction (no look-ahead), purged CV splitter, CQR coverage
  on synthetic data, Deflated Sharpe math, and artifact schema. Network tests marked/skipped in CI.

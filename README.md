# StockPredictionLE

A €0/month probabilistic stock-return forecasting system for the AEX
(Amsterdam) universe. It fetches price and macroeconomic data, engineers
features, fits quantile models, calibrates them against historical error,
and serves the results through a Streamlit dashboard.

**This is an educational decision-support tool, not financial advice.**
Read the [Caveats](#caveats) section before drawing any conclusions from
the numbers it produces.

## What this is (and isn't)

StockPredictionLE predicts a *distribution* of next-month returns per stock
— five quantiles (5/25/50/75/95%) — rather than a single point forecast, and
is explicit about how uncertain those distributions are. The honest
expectation, consistent with the finance-forecasting literature, is that
monthly out-of-sample R² for individual equity returns sits within about
±0.5% of zero. That is not a bug: monthly stock returns are close to a
random walk, and any system claiming to reliably beat that with a public,
free-data, single-model pipeline should be treated with suspicion. The
value of this project is in **calibrated uncertainty** (a 90% interval
that actually covers ~90% of outcomes) and an **honest backtest** (Sharpe
ratios reported after costs and after being deflated for the number of
strategy variants effectively trialed), not in alpha generation.

See the [design doc](docs/superpowers/specs/2026-08-26-stock-prediction-system-design.md)
for the original design decisions and rationale.

## Quickstart

```bash
# 1. Clone and set up the environment (Python >= 3.11, managed with uv)
git clone <this-repo-url>
cd StockPredictionLE
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Run the full pipeline: fetch data, train, calibrate, backtest, write artifacts
python -m stockpred run

# 3. Launch the dashboard (reads only artifacts/, no live network calls)
streamlit run app/streamlit_app.py
```

Running the pipeline the first time fetches ~25 years of daily prices for
24 tickers plus a dozen macro series — expect it to take several minutes
(mostly walk-forward LightGBM training and per-ticker GARCH simulation).
Subsequent runs reuse the Parquet cache under `data/raw/` unless you delete
it or the provider fetch itself is re-run.

Run `python -m stockpred fetch` to refresh the data cache without
retraining, or `python -m stockpred run --skip-fetch` to retrain against
whatever is already cached.

## Architecture

```
 ┌─────────────┐   ┌──────────────────┐   ┌───────────────────────┐
 │  Data layer  │──▶│ Features + PCA   │──▶│ LightGBM quantiles    │
 │ yfinance     │   │ momentum, vol,   │   │ (q05/q25/q50/q75/q95) │
 │ FRED, ECB,   │   │ Amihud, beta,    │   │ purged/embargoed      │
 │ Fama-French  │   │ 5 macro factors  │   │ walk-forward CV       │
 └─────────────┘   └──────────────────┘   └───────────┬───────────┘
                                                        │
                            ┌───────────────────────────┴────────────┐
                            ▼                                        ▼
                  ┌───────────────────┐                   ┌──────────────────┐
                  │ GJR-GARCH(1,1,1)   │                   │ Conformalized     │
                  │ skew-t simulation  │                   │ Quantile Regr.    │
                  │ (per-stock vol +   │                   │ (CQR, split-      │
                  │  return quantiles) │                   │  conformal)       │
                  └─────────┬──────────┘                   └────────┬─────────┘
                            └───────────────┬───────────────────────┘
                                            ▼
                                ┌────────────────────────┐
                                │ Ensemble + artifacts    │
                                │ forecasts.parquet,      │
                                │ diagnostics.json,       │
                                │ backtest.json,          │
                                │ manifest.json           │
                                └───────────┬─────────────┘
                                            ▼
                                ┌────────────────────────┐
                                │ Streamlit dashboard     │
                                │ (reads artifacts only)  │
                                └────────────────────────┘
```

1. **Data layer** (`stockpred/data/`) — prices via yfinance
   (`<TICKER>.AS`), macro via FRED's keyless CSV endpoint + ECB
   (`ecbdata`), Fama-French Europe factors via `pandas-datareader`. Raw
   pulls are cached as Parquet under `data/raw/`; a DuckDB database
   (`data/stockpred.duckdb`) provides SQL views over that cache.
2. **Features** (`stockpred/features/`) — per-stock momentum (1/3/6/12m,
   12-1), realized volatility (1/3/12m), Amihud illiquidity, dollar
   volume, rolling market beta; a macro panel reduced to 5 PCA factors
   (Stock-Watson/FAVAR-lite), refit on train-fold-only months every walk-
   forward fold so no validation/test information leaks into the factors.
3. **Workhorse model** (`stockpred/models/workhorse.py`) — one LightGBM
   quantile regressor per quantile level, monthly horizon, exponential
   recency-decay sample weighting (~5-year half-life). Evaluated with
   **purged, embargoed walk-forward cross-validation**
   (`stockpred/validation.py`): test blocks never overlap train blocks,
   and an embargo gap removes training rows whose 1-month-ahead target
   would otherwise leak across the train/test boundary.
4. **Calibration** — GJR-GARCH(1,1,1) with skew-t innovations
   (`stockpred/models/garch.py`) fit per stock on trailing daily returns,
   simulated forward for volatility and return-quantile bands; Conformalized
   Quantile Regression (`stockpred/models/conformal.py`, split-conformal on
   a held-out calibration window) widens or shrinks the LightGBM quantile
   bands so their empirical coverage matches their nominal level.
5. **Backtest** (`stockpred/backtest.py`) — a simple long/cash monthly
   threshold rule, DEGIRO-like trading costs, and the **Deflated Sharpe
   Ratio** (Bailey & López de Prado 2014), which discounts the raw Sharpe
   ratio for the number of strategy variants effectively trialed across
   walk-forward folds — the honest way to ask "is this luck?"
6. **Pipeline** (`stockpred/pipeline.py`, CLI `python -m stockpred`) —
   `fetch → refresh_duckdb → features → walk_forward → calibration →
   backtest → garch → production_forecast → assemble_and_write`. A GitHub
   Actions weekly cron (`.github/workflows/weekly.yml`) runs this end to
   end and commits refreshed `artifacts/` back to the repo.
7. **Dashboard** (`app/streamlit_app.py`, `app/charts.py`) — reads only
   `artifacts/*.parquet|json`, no live network calls, so it stays
   lightweight enough for Streamlit Community Cloud. Three pages:
   per-stock fan charts (Plotly) with a 12-month "humble" scenario band,
   a diagnostics page (R² by fold, coverage before/after calibration,
   pinball loss), and a backtest page (equity curve, Sharpe/PSR/DSR,
   drawdown, trade log).

## Provider notes

- **yfinance** is the primary and only price provider. **Stooq is not
  used**: pandas-datareader 0.11.1 removed its Stooq module, and Stooq's
  own CSV endpoint sits behind a JS anti-bot challenge that isn't
  practical to solve here (see `stockpred/data/prices.py`). If yfinance
  fails for a ticker after 3 retries, the pipeline falls back to that
  ticker's last cached Parquet snapshot.
- **TKWY (Just Eat Takeaway)** is excluded from `config.yaml`'s ticker
  list — it is delisted from Yahoo Finance and its `.AS` endpoint
  verifiably 404s. The universe is 24 AEX tickers, not 25.
- **FRED** works keyless via the `fredgraph.csv` endpoint by default (set
  `FRED_API_KEY` to use `fredapi` instead, not required). FRED returns
  each series' *full* history — e.g. `INDPRO` back to 1919 — so the
  pipeline clips the macro panel to `price_start` minus a 2-year buffer
  before computing which series survive the missing-data filter (see
  [Real-data findings](#real-data-findings-task-12) below).
- **ECB** (`ecbdata`) is treated as best-effort: a per-series failure is
  logged and skipped rather than aborting the fetch.
- **Fama-French Europe** (not US) factors are used via
  `pandas_datareader.data.DataReader(..., "famafrench")`, matching the
  European stock universe.

## Deploying

1. Push this repo to a **public** GitHub repository (Streamlit Community
   Cloud's free tier requires a public repo, or a connected private repo
   on a paid plan).
2. The weekly cron in `.github/workflows/weekly.yml` runs every Monday at
   06:00 UTC (`workflow_dispatch` also lets you trigger it manually from
   the Actions tab). It installs the package, runs
   `python -m stockpred run`, and commits any changed files under
   `artifacts/` back to the repo as `github-actions[bot]`. No secrets are
   required — every provider used here is keyless.
3. On [Streamlit Community Cloud](https://streamlit.io/cloud), create a
   new app pointing at this repo, branch `main` (or whichever branch you
   deploy from), and entrypoint `app/streamlit_app.py`. Streamlit Cloud
   installs from `requirements.txt` (deliberately lightweight — no
   lightgbm/arch/scikit-learn — since the dashboard only reads committed
   artifacts).
4. **Keep-alive note:** Streamlit Community Cloud puts idle-owner apps to
   sleep, and GitHub disables a scheduled Actions workflow after **60 days**
   with no commits to the repository. The weekly cron's own commits reset
   that clock every week it runs successfully — but if you fork this,
   pause the cron, or the cron itself starts silently failing (e.g. a
   provider outage with no cache to fall back to), make sure *something*
   commits to the repo at least once every 60 days or GitHub will disable
   the schedule and you'll need to manually re-enable it from the Actions
   tab.

## Results snapshot (first real-data run, 2026-08-26)

From the first full run against real providers (24 AEX tickers, 2000–2026
daily prices, 14 macro series, walk-forward LightGBM + CQR + GJR-GARCH):

| Metric | Value | Note |
|---|---|---|
| Tickers fetched / failed | 24 / 0 | all resolve after TKWY's removal |
| Pooled OOS R² (median forecast vs historical-mean benchmark) | −3.7% | negative, i.e. *below* the naive benchmark; per-fold range −22.6% to +9.6% — consistent with "no detectable edge," not leakage (leakage would show up as R² well above zero, not below it) |
| Raw 90% interval coverage (pre-calibration) | 76.7% | under-covers, as expected for an uncalibrated quantile regressor |
| Calibrated 90% interval coverage (post-CQR) | 89.6% | inside the 87–93% target band |
| Backtest: annualized return / vol / Sharpe | 0.75% / 11.3% / 0.12 | net of DEGIRO-like costs |
| Backtest: PSR / DSR | 0.63 / 0.13 | DSR (deflated for 8 walk-forward-fold trials) does *not* support a genuine-skill claim |
| Buy-and-hold benchmark total return (same period) | +205% | the strategy's conservative long/cash rule captures very little of the AEX's 2018–2026 bull run — an honest cost of caution, not a bug |

Full per-fold breakdowns, pinball loss, monthly-return series, and stage
timings are in
`.superpowers/sdd/2026-08-26-stock-prediction-system/task-12-report.md`.

## Real-data findings (Task 12)

Findings from the first full real-provider run, kept here for anyone
re-running this pipeline against fresh data:

- **TKWY delisted.** Removed from `config.yaml`; see Provider notes above.
- **Macro history clipping.** FRED's keyless endpoint returns each
  series' entire history. Left unclipped, `build_macro_wide`'s wide panel
  is forced back to `INDPRO`'s 1919 start date, which makes every
  later-starting series (`VIXCLS` from 1990, `DTWEXBGS` from 2006, ...)
  read as mostly-missing over that artificially long range and get
  dropped by the (correct, intentional) >30%-missing column filter — only
  3 of 14 configured macro series survived in the unclipped run, capping
  the PCA factor extractor below the configured `n_pca_factors=5` and
  crashing the pipeline downstream (`KeyError: ['F4', 'F5'] not in
  index`, since feature-column lists elsewhere assume the configured
  factor count). Fixed by clipping the macro panel to `price_start` minus
  a 2-year buffer (room for the 12-month log-diff lookback) before
  computing the missing-data filter (`stockpred/features/macro_pca.py`'s
  new `min_date` parameter, wired up in `stockpred/pipeline.py`) — see
  the full metrics dump in the task report for which series survive after
  the fix.
- Full metrics (R² per fold, coverage, pinball loss, backtest DSR/PSR,
  stage timings, ticker fetch results) are in
  `.superpowers/sdd/2026-08-26-stock-prediction-system/task-12-report.md`.

## Caveats

- **The edge, if any, is small and fragile.** Free daily OHLCV and public
  macro series carry real noise and reporting lag; a single-model,
  monthly-horizon pipeline trained on ~25 years of data for 24 stocks is
  not going to reliably beat a random walk. Treat any positive backtest
  number with skepticism proportional to how good it looks.
- **Overfitting is a constant danger**, not a one-time risk to check off.
  Walk-forward CV with purging and embargo, a held-out conformal
  calibration window, and the Deflated Sharpe Ratio's multiple-testing
  correction all exist specifically to catch it — but no amount of
  validation methodology makes a small, noisy dataset immune to spurious
  patterns.
- **Free data has real quality limits**: yfinance can silently stop
  updating a ticker, adjust historical prices retroactively, or return
  gappy data around corporate actions; FRED/ECB series get revised after
  publication. The pipeline degrades to cached data on failure rather
  than crashing, but "degrades gracefully" is not the same as "always
  correct."
- **The 12-month scenario bands are wide, humble ranges built from GARCH
  simulation, not predictions.** They express plausible outcome dispersion
  a year out, not a forecast of where a stock will be. Treat a "50%
  chance the return is inside this ±20 percentage-point band" claim as
  exactly that — a statement about width, not precision.
- **This is not financial advice.** The dashboard displays this warning
  on every page. Nothing here accounts for your personal financial
  situation, risk tolerance, or tax circumstances, and past OOS
  performance — even honestly measured — is not a guarantee of anything
  about future performance.

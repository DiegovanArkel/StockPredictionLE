# StockPredictionLE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A €0/month probabilistic AEX stock forecasting system: factor-augmented LightGBM quantile model + GARCH/conformal calibration, weekly-refreshable pipeline writing Parquet/JSON artifacts, and a Streamlit fan-chart dashboard reading only those artifacts.

**Architecture:** A single Python package `stockpred` with layered modules (data → features → models → calibration → backtest → artifacts) orchestrated by a CLI pipeline; the Streamlit app is a pure consumer of `artifacts/`. All data pulls cache to Parquet; DuckDB provides queryable views. PCA and all model fitting happen inside walk-forward folds to prevent leakage.

**Tech Stack:** Python 3.11 (uv-managed venv at `.venv`), pandas, numpy, scikit-learn, lightgbm, arch, yfinance, pandas-datareader (Fama-French only), duckdb, pyarrow, streamlit, plotly, pytest, pyyaml, requests.

## Global Constraints

- Python ≥ 3.11; run everything with `.venv/bin/python` (uv venv already created at repo root).
- Price provider: yfinance primary (`.AS` suffix). Do NOT attempt to bypass Stooq's JS challenge.
- FRED: keyless `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>` by default; use `fredapi` only if `FRED_API_KEY` env var is set (optional dependency, not required).
- No look-ahead: any cross-sectional/statistical fit (PCA, scalers, conformal offsets) is fit on training data only within each fold.
- All returns stored as decimal fractions (0.05 = 5%), never percent.
- Monthly period timestamps are month-END (`pd.Timestamp` at month end).
- Every artifact write updates `artifacts/manifest.json` with timestamps, row counts, data staleness, and failed tickers.
- Network-dependent tests marked `@pytest.mark.network` and excluded by default (`-m "not network"`).
- Educational tool, not financial advice: dashboard must show this caveat prominently.
- Commit after each task with a conventional-commit message.

---

### Task 1: Project scaffold + config

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `config.yaml`, `stockpred/__init__.py`, `stockpred/config.py`, `tests/test_config.py`, `tests/conftest.py`, `pytest.ini`

**Interfaces:**
- Produces: `stockpred.config.load_config(path: str | Path = "config.yaml") -> Config` where `Config` is a frozen dataclass with attributes:
  - `tickers: list[str]` (bare AEX names, e.g. "ASML", "ADYEN"), `yf_suffix: str = ".AS"`
  - `price_start: str` ("2000-01-01"), `fred_series: list[str]`, `ecb_series: list[str]`
  - `n_pca_factors: int = 5`, `quantiles: list[float] = [0.05,0.25,0.5,0.75,0.95]`
  - `horizon_months: int = 1`, `decay_half_life_years: float = 5.0`
  - `n_folds: int = 8`, `embargo_months: int = 1`, `test_window_months: int = 12`
  - `cost_fixed_eur: float = 2.0`, `cost_bps: float = 2.61`, `signal_threshold: float = 0.01`, `loss_tolerance: float = -0.10`
  - `data_dir: Path`, `artifacts_dir: Path`, `duckdb_path: Path`
- `config.yaml` holds the current AEX-25 universe: ABN, ADYEN, AGN, AD, AKZA, ASM, ASML, ASRNL, BESI, DSFIR, EXO, HEIA, IMCD, INGA, KPN, MT, NN, PHIA, PRX, RAND, REN, SHELL, TKWY, UMG, WKL. FRED series: INDPRO, CPIAUCSL, UNRATE, FEDFUNDS, T10Y2Y, T10Y3M, VIXCLS, DTWEXBGS, DGS10, DCOILBRENTEU, USEPUINDXD (use monthly resample last), M2SL. ECB series (ecbdata keys): `ICP.M.U2.N.000000.4.ANR` (HICP inflation), `MIR.M.U2.B.A2A.A.R.A.2240.EUR.N` (lending rate) — treat ECB as best-effort optional (skip on failure with warning).
- pyproject: project name `stockpred`, deps as in Tech Stack; `[tool.pytest.ini_options]` NOT used (use pytest.ini with `addopts = -m "not network"` and `markers = network`).
- .gitignore: `.venv/`, `__pycache__/`, `data/raw/`, `data/*.duckdb*`, `.pytest_cache/`, `*.egg-info/`. Note: `artifacts/` is COMMITTED (dashboard reads it from the repo).

**Steps:**
- [ ] Write `tests/test_config.py`: `load_config()` returns Config with 25 tickers, 5 quantiles summing pattern, paths are `Path` objects, unknown YAML keys raise `ValueError`.
- [ ] Run tests, verify failure. Implement `config.py` (dataclass + yaml load + validation). Verify pass.
- [ ] `uv pip install -p .venv/bin/python -e ".[dev]"` works; `pytest` green.
- [ ] Commit `feat: project scaffold and config`.

### Task 2: Data providers + storage

**Files:**
- Create: `stockpred/data/__init__.py`, `stockpred/data/prices.py`, `stockpred/data/macro.py`, `stockpred/data/factors.py`, `stockpred/data/storage.py`, `tests/test_storage.py`, `tests/test_providers.py`

**Interfaces:**
- Produces:
  - `prices.fetch_prices(tickers, start, end=None, yf_suffix=".AS") -> tuple[pd.DataFrame, list[str]]` — tidy df `[date, ticker, close, adj_close, volume]` (ticker WITHOUT suffix), plus list of failed tickers. Uses yfinance `download` per-ticker with 3 retries/backoff (sleep 2·attempt), `auto_adjust=False` to keep both close and adj close.
  - `macro.fetch_fred(series_ids) -> pd.DataFrame` tidy `[date, series_id, value]`, keyless CSV endpoint via requests (User-Agent header), per-series try/except collecting failures; `macro.fetch_ecb(series_keys) -> pd.DataFrame` same shape via `ecbdata.ECB_Data_Portal` — wrap entire call in try/except returning empty df on failure (best-effort).
  - `factors.fetch_famafrench(start="2000-01-01") -> pd.DataFrame` — monthly `[date, mkt_rf, smb, hml, mom, rf]` in decimals, month-end dates; joins pandas-datareader `Europe_3_Factors` table 0 with `Europe_Mom_Factor` table 0.
  - `storage.save_parquet(df, name, data_dir)` / `storage.load_parquet(name, data_dir) -> pd.DataFrame | None` (None if missing); `storage.refresh_duckdb(duckdb_path, data_dir)` creating/replacing DuckDB views over the parquet files (`prices`, `macro`, `factors`).
  - `storage.cached_fetch(name, fetch_fn, data_dir) -> tuple[pd.DataFrame, bool]` — calls fetch_fn; on any exception or empty result loads cache and returns `(cached_df, True /*stale*/)`; on success saves cache and returns `(df, False)`. Raises `RuntimeError` if fetch fails AND no cache.
- Consumes: `Config` from Task 1.

**Steps:**
- [ ] Tests first (offline): `cached_fetch` fallback logic with a failing fetch_fn + pre-seeded cache; parquet round-trip; duckdb views queryable via `duckdb.connect`. Provider smoke tests marked `@pytest.mark.network`.
- [ ] Implement; offline tests green.
- [ ] Run network tests once manually (`pytest -m network`) — expect ≥20/25 AEX tickers resolve; record failures.
- [ ] Commit `feat: data providers with cache fallback and duckdb storage`.

### Task 3: Feature engineering (monthly panel + macro PCA)

**Files:**
- Create: `stockpred/features/__init__.py`, `stockpred/features/stock.py`, `stockpred/features/macro_pca.py`, `tests/test_features.py`, `tests/test_macro_pca.py`

**Interfaces:**
- Produces:
  - `stock.build_monthly_panel(prices: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame` — index reset tidy frame, one row per (ticker, month_end date), columns: `ret_1m` (that month's simple return from adj_close), features known at month end: `mom_1m, mom_3m, mom_6m, mom_12m, mom_12_1` (12m return skipping most recent month), `vol_1m, vol_3m, vol_12m` (annualized std of daily returns), `amihud` (monthly mean |ret|/(close·volume), log1p-scaled), `dollar_vol` (log mean close·volume), `beta_12m` (rolling regression of daily stock returns on daily-resampled mkt_rf — approximate via monthly returns over 36m window, min 12), and target `fwd_ret_1m = ret_1m` shifted -1 within ticker. Drop rows with missing target; features may have NaN (LightGBM handles).
  - `macro_pca.build_macro_wide(macro: pd.DataFrame) -> pd.DataFrame` — pivot to monthly wide (resample('ME').last()), transform each series to stationarity: for strictly-positive level series (INDPRO, CPIAUCSL, M2SL) use 12m log-diff; for rates/spreads/indices (UNRATE, FEDFUNDS, T10Y2Y, T10Y3M, VIXCLS, DGS10, DTWEXBGS, DCOILBRENTEU, USEPUINDXD, ECB series) use 1m diff; forward-fill ≤2 months, drop columns >30% missing.
  - `macro_pca.MacroFactorExtractor(n_factors=5)` with `.fit(macro_wide_train)` (standardize + sklearn PCA) and `.transform(macro_wide) -> pd.DataFrame` columns `F1..F5` indexed by date. NEVER fit on full sample in production paths.
- Consumes: tidy prices/macro/factors frames from Task 2.

**Steps:**
- [ ] Tests with synthetic data: momentum math exact on constructed series; `fwd_ret_1m` at month t equals `ret_1m` at t+1 (no-lookahead test); mom_12_1 excludes last month; PCA explains >90% variance on rank-3 synthetic panel; transform on unseen dates works.
- [ ] Implement; green; commit `feat: monthly stock features and macro PCA factors`.

### Task 4: Validation utilities (purged walk-forward, decay weights, deflated Sharpe)

**Files:**
- Create: `stockpred/validation.py`, `tests/test_validation.py`

**Interfaces:**
- Produces:
  - `purged_walk_forward(dates: pd.Series, n_folds, test_window_months, embargo_months, min_train_months=120) -> Iterator[tuple[np.ndarray, np.ndarray]]` — expanding-window splits over unique sorted month-end dates: each fold's test block is `test_window_months` consecutive months ending at the sample end walking backward; train = all months ≤ (test_start − embargo − horizon buffer of 1 month). Returns positional index arrays into the original `dates` series. Skips folds with < min_train_months.
  - `decay_weights(dates, half_life_years) -> np.ndarray` — exponential 0.5**(age_years/half_life), normalized mean 1.
  - `deflated_sharpe_ratio(returns: np.ndarray, n_trials: int, periods_per_year=12) -> dict` with keys `sharpe, psr, dsr` per Bailey & López de Prado (expected max Sharpe under n_trials using Euler–Mascheroni approximation; PSR with skew/kurtosis adjustment).
- Consumes: nothing beyond pandas/numpy/scipy.

**Steps:**
- [ ] Tests: no train date within embargo of test start; folds don't overlap test months; weights halve at half-life; DSR of pure-noise returns ≈ low (<0.5 psr on average, deterministic seed); DSR of strong-signal series high (>0.9).
- [ ] Implement; green; commit `feat: purged walk-forward CV, decay weights, deflated Sharpe`.

### Task 5: Workhorse LightGBM quantile model + walk-forward evaluation

**Files:**
- Create: `stockpred/models/__init__.py`, `stockpred/models/workhorse.py`, `tests/test_workhorse.py`

**Interfaces:**
- Produces:
  - `WorkhorseModel(quantiles, lgb_params=None)` — `.fit(X: pd.DataFrame, y, sample_weight=None)` trains one LGBMRegressor(objective="quantile", alpha=q) per q with defaults `n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0`; `.predict(X) -> pd.DataFrame` columns `q05,q25,q50,q75,q95` (names = f"q{int(q*100):02d}"), rows aligned to X index, with non-crossing enforced by sorting each row.
  - `run_walk_forward(panel, macro_wide, cfg) -> WalkForwardResult` dataclass: `oos: pd.DataFrame` (per ticker/date: all quantile cols, `y_true`, `fold`), `metrics: dict` (per-fold and pooled: `r2_oos_median` vs historical-mean benchmark [benchmark = expanding mean of training `fwd_ret_1m` pooled], `pinball_loss` per quantile, `coverage_90`), and `feature_importance: pd.DataFrame`. Inside each fold: fit `MacroFactorExtractor` on train months only, merge F1..F5 onto panel by month, fit model with decay weights, predict test block. Also retains `calibration` split: last 24 train months held out per fold, with predictions stored in `cal: pd.DataFrame` (same schema as oos) for Task 7.
  - Feature list constant `FEATURE_COLS` exported (stock features + F1..F5).
- Consumes: Tasks 3, 4 interfaces exactly as defined.

**Steps:**
- [ ] Tests on synthetic panel (~30 tickers × 200 months, y = 0.3·signal + noise): quantiles non-crossing; q50 beats naive mean benchmark on synthetic signal (r2_oos_median > 0); no fold trains on data ≥ test_start − embargo; cal frames disjoint from oos.
- [ ] Implement; green (allow ~2 min runtime); commit `feat: LightGBM quantile workhorse with leak-safe walk-forward`.

### Task 6: GARCH layer

**Files:**
- Create: `stockpred/models/garch.py`, `tests/test_garch.py`

**Interfaces:**
- Produces:
  - `fit_and_simulate(daily_returns: pd.Series, horizon_days=21, n_sims=5000, seed=0) -> GarchResult` dataclass: `ann_vol_forecast: float`, `month_quantiles: dict[str, float]` (keys q05..q95 of simulated compounded h-day returns), `year_quantiles: dict[str, float]` (h=252 simulation), `converged: bool`. Fits GJR-GARCH(1,1,1) with skew-t (`arch_model(returns*100, vol="GARCH", p=1, o=1, q=1, dist="skewt")`), scales back to decimals; simulation via `res.forecast(horizon=..., method="simulation", simulations=n_sims)` compounding simulated returns. On convergence failure or <500 daily obs: fall back to empirical bootstrap quantiles of daily returns (compounded), `converged=False`.
  - `garch_all(prices: pd.DataFrame, lookback_years=8) -> pd.DataFrame` — one row per ticker with the above fields flattened (`ticker, ann_vol, m_q05..m_q95, y_q05..y_q95, converged`).
- Consumes: tidy prices from Task 2.

**Steps:**
- [ ] Tests: on simulated GARCH data recovered ann_vol within factor 2; bootstrap fallback triggers on 100-obs series; quantiles monotone; deterministic under seed.
- [ ] Implement; green; commit `feat: GJR-GARCH skew-t simulation layer with bootstrap fallback`.

### Task 7: Conformal calibration (CQR)

**Files:**
- Create: `stockpred/models/conformal.py`, `tests/test_conformal.py`

**Interfaces:**
- Produces:
  - `cqr_offsets(cal_lo, cal_hi, cal_y, alpha) -> float` — split-CQR conformity score `E_i = max(cal_lo−y, y−cal_hi)`; returns the `ceil((n+1)(1−alpha))/n` empirical quantile of E.
  - `apply_cqr(pred: pd.DataFrame, offsets: dict) -> pd.DataFrame` — widens (q05,q95) by `offsets["90"]` and (q25,q75) by `offsets["50"]`, adds columns `q05_cal,q25_cal,q75_cal,q95_cal` (q50 unchanged, re-sorted for non-crossing).
  - `calibrate_from_wf(wf: WalkForwardResult) -> tuple[dict, pd.DataFrame]` — computes offsets from pooled `wf.cal`, applies to `wf.oos`, returns (offsets, oos_calibrated) plus empirical coverage before/after stored in `offsets["coverage_raw_90"], offsets["coverage_cal_90"]`.
- Consumes: `WalkForwardResult` from Task 5.

**Steps:**
- [ ] Tests: on synthetic data with deliberately narrow intervals (coverage 70%), CQR restores 90% ±3pp on a fresh test set; offsets ≥ 0 in that case; degenerate n<20 raises ValueError.
- [ ] Implement; green; commit `feat: conformalized quantile regression calibration`.

### Task 8: Forecast assembly + artifact writer

**Files:**
- Create: `stockpred/artifacts.py`, `tests/test_artifacts.py`

**Interfaces:**
- Produces:
  - `assemble_forecasts(final_pred, garch_df, offsets, as_of) -> pd.DataFrame` — per ticker: monthly cols `q05..q95` = CQR-calibrated LightGBM quantiles; `ens_q*` = 50/50 average of calibrated LightGBM and GARCH month quantiles (median: average of q50s); yearly cols `y_q05..y_q95` from GARCH year quantiles re-centered on 12·ens_q50 (humble wide bands); plus `ann_vol`, `garch_converged`.
  - `write_artifacts(artifacts_dir, forecasts, wf_metrics, offsets, backtest_summary, history, failures, staleness) -> None` — writes `forecasts.parquet`, `oos_history.parquet` (calibrated OOS predictions + y_true for the diagnostics page), `backtest.json`, `diagnostics.json` (wf metrics + coverage + offsets), `manifest.json` (`generated_at` ISO, row counts, failed tickers, stale sources, package versions).
  - `load_artifacts(artifacts_dir) -> dict` with keys `forecasts, oos_history, diagnostics, backtest, manifest` (DataFrames/dicts) — the ONLY API the dashboard uses.
- Consumes: Tasks 5–7 outputs.

**Steps:**
- [ ] Tests: round-trip write/load; manifest fields present; ensemble math exact on toy inputs; yearly bands wider than monthly.
- [ ] Implement; green; commit `feat: forecast assembly and artifact read/write layer`.

### Task 9: Backtest with deflated Sharpe

**Files:**
- Create: `stockpred/backtest.py`, `tests/test_backtest.py`

**Interfaces:**
- Produces:
  - `run_backtest(oos_calibrated: pd.DataFrame, cfg) -> dict` — decision rule per month: equal-weight long every ticker with `q50 > cfg.signal_threshold` and `q05_cal > cfg.loss_tolerance`; hold cash (0%) otherwise. Portfolio monthly return = mean of selected tickers' `y_true` minus costs: per position entered/exited, cost = `cost_bps/1e4 + cost_fixed_eur/position_notional` with €10,000 initial capital equal-split. Returns dict: `monthly_returns` (list of {date, ret, n_positions}), `total_return, ann_return, ann_vol, sharpe, dsr, psr, max_drawdown, n_trades, benchmark_total_return` (buy-and-hold equal-weight all tickers), `per_fold_sharpe` (list).
  - Uses `deflated_sharpe_ratio` with `n_trials=cfg.n_folds` (honest accounting of selection over folds).
- Consumes: calibrated OOS frame (Task 7), `deflated_sharpe_ratio` (Task 4).

**Steps:**
- [ ] Tests: zero-cost, all-signals-pass case equals equal-weight mean return; costs reduce returns; no-signal months are flat; max_drawdown correct on constructed path.
- [ ] Implement; green; commit `feat: threshold decision-rule backtest with deflated Sharpe`.

### Task 10: Pipeline CLI + GitHub Actions

**Files:**
- Create: `stockpred/pipeline.py`, `stockpred/__main__.py`, `.github/workflows/weekly.yml`, `tests/test_pipeline.py`

**Interfaces:**
- Produces: `python -m stockpred run [--config config.yaml] [--skip-fetch]` executing: fetch (cached_fetch per source) → duckdb refresh → features → walk-forward → CQR → GARCH → final-model refit on ALL data (with full-sample-fitted MacroFactorExtractor — allowed ONLY for the production forecast, never for reported metrics) → assemble → backtest → write_artifacts. Also `python -m stockpred fetch` (data only). Log progress per stage with timings.
- Workflow: cron `0 6 * * 1`, ubuntu-latest, setup-python 3.11, `pip install -e .`, `python -m stockpred run`, commit `artifacts/` back with `github-actions[bot]` — include `permissions: contents: write` and a guard so it still succeeds when nothing changed.
- Consumes: everything above.

**Steps:**
- [ ] Test: end-to-end `run(--skip-fetch)` on small synthetic parquet fixtures produces all artifact files.
- [ ] Implement; green; commit `feat: pipeline CLI and weekly GitHub Actions cron`.

### Task 11: Streamlit dashboard

**Files:**
- Create: `app/streamlit_app.py`, `app/charts.py`, `requirements.txt` (for Streamlit Cloud: streamlit, plotly, pandas, pyarrow)

**Interfaces:**
- Consumes: ONLY `stockpred.artifacts.load_artifacts` semantics — but implement standalone reads in `app/` (no heavy deps import) reading `artifacts/*.parquet|json` directly so Streamlit Cloud needs no lightgbm/arch install.
- Produces: pages via sidebar: **Forecasts** (stock selector; fan chart: last 24m of actual cumulative price indexed to 100, fan of monthly quantile bands projected 1 month, plus a yearly humble-band panel; table of q05..q95 monthly and yearly; ensemble toggle), **Diagnostics** (OOS R² vs benchmark per fold bar chart, coverage gauge raw vs calibrated, pinball losses, feature importances if present), **Backtest** (equity curve vs buy-hold benchmark from monthly_returns, Sharpe/DSR/PSR/max-DD stat row, trade count), persistent caveat banner: "Educational tool — not financial advice. Expected edge after costs is likely zero." Fan charts: plotly filled bands (5–95 light, 25–75 darker, median line), dark-mode friendly default template.

**Steps:**
- [ ] Build with sample artifacts from Task 10's test run; verify `streamlit run app/streamlit_app.py` renders without exceptions (headless smoke: `python -c "import app.streamlit_app"` guarded by `st.runtime.exists()` not required — use function-based layout and a `main()`).
- [ ] Commit `feat: streamlit fan-chart dashboard reading precomputed artifacts`.

### Task 12: Full real-data integration run + README

**Files:**
- Modify: `README.md`; artifacts under `artifacts/`

**Steps:**
- [ ] Run `python -m stockpred run` against real providers. Fix any real-world data issues (ticker failures, NaN handling).
- [ ] Verify benchmarks: (a) data refresh completes with zero manual steps; (b) report OOS R² vs historical mean (expect ≈0±0.5%; if >2% suspect leakage and re-audit embargo); (c) calibrated 90% coverage within 87–93%; (d) backtest DSR reported honestly.
- [ ] README: quickstart, architecture diagram (text), provider notes (Stooq status), deploy instructions (GitHub repo + Actions + Streamlit Cloud), 60-day commit keep-alive note, caveats.
- [ ] Launch dashboard locally, visually verify fan charts, screenshot for report.
- [ ] Commit `feat: first real-data artifact build and documentation`.

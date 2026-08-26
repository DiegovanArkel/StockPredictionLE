"""Smoke tests for the Streamlit dashboard (Task 11), using Streamlit's
``AppTest`` harness (``streamlit.testing.v1``).

These do not exercise app/streamlit_app.py's *content* in depth -- they
assert the three pages render without raising, both against a populated
sample artifacts dir and against a missing/empty one (a fresh clone before
the pipeline has ever run, per the artifact contract in
stockpred/artifacts.py: a missing file maps to ``None`` and the dashboard
must show a placeholder, not crash).

Sample artifacts are built directly with ``stockpred.artifacts.write_artifacts``
(this test file is allowed to import stockpred; app/ itself is not -- see
the module docstring in app/streamlit_app.py).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from stockpred.artifacts import write_artifacts

APP_PATH = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
_PAGES = ["Forecasts", "Diagnostics", "Backtest"]


def _write_sample_artifacts(artifacts_dir: Path) -> None:
    rng = np.random.default_rng(7)
    tickers = ["AAA", "BBB", "CCC"]
    as_of = date(2026, 8, 24)

    fc_rows = []
    for i, t in enumerate(tickers):
        q05, q25, q50, q75, q95 = sorted(rng.normal(0, 0.05, 5))
        y_q05, y_q25, y_q50, y_q75, y_q95 = sorted(rng.normal(0, 0.2, 5))
        pred_date = as_of - timedelta(days=45) if i == 0 else as_of
        fc_rows.append(
            dict(
                ticker=t, as_of=as_of.isoformat(), pred_date=pred_date.isoformat(),
                q05=q05, q25=q25, q50=q50, q75=q75, q95=q95,
                raw_q05=q05 - 0.01, raw_q25=q25 - 0.005, raw_q75=q75 + 0.005, raw_q95=q95 + 0.01,
                ens_q05=q05, ens_q25=q25, ens_q50=q50, ens_q75=q75, ens_q95=q95,
                y_q05=y_q05, y_q25=y_q25, y_q50=y_q50, y_q75=y_q75, y_q95=y_q95,
                ann_vol=0.25, garch_converged=True,
            )
        )
    forecasts = pd.DataFrame(fc_rows)

    hist_rows = []
    months = pd.date_range(end=pd.Timestamp(as_of) - pd.DateOffset(months=1), periods=6, freq="MS")
    for t in tickers:
        for j, m in enumerate(months):
            hist_rows.append(
                dict(
                    ticker=t, date=m, q05=-0.05, q25=-0.02, q50=0.01, q75=0.03, q95=0.06,
                    y_true=float(rng.normal(0.01, 0.05)), fold=j // 3,
                    q05_cal=-0.055, q25_cal=-0.022, q75_cal=0.032, q95_cal=0.065,
                )
            )
    oos_history = pd.DataFrame(hist_rows)

    wf_metrics = {
        "pooled": {
            "r2_oos_median": 0.02,
            "pinball_loss": {k: 0.03 for k in ["q05", "q25", "q50", "q75", "q95"]},
            "coverage_90": 0.89,
        },
        "folds": {
            0: {"r2_oos_median": 0.01, "pinball_loss": {k: 0.03 for k in ["q05", "q25", "q50", "q75", "q95"]}, "coverage_90": 0.87, "n_train_months": 60, "n_test_rows": 30},
            1: {"r2_oos_median": 0.03, "pinball_loss": {k: 0.03 for k in ["q05", "q25", "q50", "q75", "q95"]}, "coverage_90": 0.91, "n_train_months": 66, "n_test_rows": 30},
        },
    }
    offsets = {"90": 0.01, "50": 0.005, "coverage_raw_90": 0.82, "coverage_cal_90": 0.89}

    monthly_returns = [
        {"date": m.strftime("%Y-%m-%d"), "ret": 0.01, "gross_ret": 0.012, "n_positions": 2, "cost_eur": 3.0}
        for m in months
    ]
    net = np.array([m["ret"] for m in monthly_returns])
    total_return = float(np.prod(1 + net) - 1)
    backtest_summary = {
        "monthly_returns": monthly_returns,
        "total_return": total_return,
        "ann_return": 0.05,
        "ann_vol": 0.1,
        "sharpe": 0.5,
        "psr": 0.6,
        "dsr": 0.4,
        "max_drawdown": -0.08,
        "n_trades": 12,
        "benchmark_total_return": 0.07,
        "per_fold_sharpe": [{"fold": 0, "sharpe": 0.3}, {"fold": 1, "sharpe": 0.6}],
        "n_months": len(months),
        "final_capital": 10_000.0 * (1 + total_return),
    }

    write_artifacts(
        artifacts_dir,
        forecasts=forecasts,
        wf_metrics=wf_metrics,
        offsets=offsets,
        backtest_summary=backtest_summary,
        history=oos_history,
        failures=[],
        staleness={},
    )


@pytest.fixture
def sample_artifacts_dir(tmp_path: Path) -> Path:
    out = tmp_path / "artifacts"
    _write_sample_artifacts(out)
    return out


def test_default_artifacts_dir_renders_without_exception(monkeypatch):
    """Repo-root artifacts/ is empty in this checkout (pre-Task-12) -- the
    default run should hit the missing-artifacts path and still render.
    """
    monkeypatch.delenv("STOCKPRED_ARTIFACTS_DIR", raising=False)
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception


def test_empty_artifacts_dir_renders_placeholders_without_exception(monkeypatch, tmp_path):
    empty_dir = tmp_path / "empty_artifacts"
    empty_dir.mkdir()
    monkeypatch.setenv("STOCKPRED_ARTIFACTS_DIR", str(empty_dir))
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception

    # Every page should show a placeholder rather than crash.
    for page in _PAGES:
        at.sidebar.radio[0].set_value(page)
        at.run()
        assert not at.exception
        assert any("No " in info.value for info in at.info)


@pytest.mark.parametrize("page", _PAGES)
def test_sample_artifacts_each_page_renders_without_exception(monkeypatch, sample_artifacts_dir, page):
    monkeypatch.setenv("STOCKPRED_ARTIFACTS_DIR", str(sample_artifacts_dir))
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception

    at.sidebar.radio[0].set_value(page)
    at.run()
    assert not at.exception


def test_forecasts_page_ensemble_toggle_renders_without_exception(monkeypatch, sample_artifacts_dir):
    monkeypatch.setenv("STOCKPRED_ARTIFACTS_DIR", str(sample_artifacts_dir))
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert not at.exception

    at.sidebar.radio[0].set_value("Forecasts")
    at.run()
    assert not at.exception

    assert len(at.checkbox) >= 1
    at.checkbox[0].set_value(True)
    at.run()
    assert not at.exception


def test_forecasts_page_stale_ticker_shows_warning(monkeypatch, sample_artifacts_dir):
    """TICK/AAA's pred_date is deliberately 45 days before as_of -- picking
    it should trigger the staleness warning (see _STALE_DAYS in
    app/streamlit_app.py).
    """
    monkeypatch.setenv("STOCKPRED_ARTIFACTS_DIR", str(sample_artifacts_dir))
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.sidebar.radio[0].set_value("Forecasts")
    at.run()

    at.selectbox[0].set_value("AAA")
    at.run()
    assert not at.exception
    assert any("Stale forecast" in e.value for e in at.error)

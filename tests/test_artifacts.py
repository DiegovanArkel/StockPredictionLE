"""Tests for stockpred.artifacts: forecast assembly + artifact writer/reader
(Task 8).

Conventions mirror tests/test_conformal.py and tests/test_garch.py: small
synthetic frames with hand-computed expected values, returns as decimal
fractions.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from stockpred.artifacts import assemble_forecasts, load_artifacts, write_artifacts


def _final_pred() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "q05": [-0.10, -0.08, -0.05],
            "q25": [-0.03, -0.02, -0.02],
            "q50": [0.01, 0.02, 0.00],
            "q75": [0.05, 0.04, 0.02],
            "q95": [0.12, 0.09, 0.05],
            "q05_cal": [-0.11, -0.085, -0.05],
            "q25_cal": [-0.035, -0.025, -0.02],
            "q75_cal": [0.055, 0.045, 0.02],
            "q95_cal": [0.13, 0.095, 0.05],
        }
    )


def _garch_df() -> pd.DataFrame:
    # BBB deliberately absent (missing-garch fallback case).
    # CCC has deliberately non-monotonic month + year quantiles to exercise
    # the per-row sort-based monotonicity fixup.
    return pd.DataFrame(
        {
            "ticker": ["AAA", "CCC"],
            "ann_vol": [0.25, 0.30],
            "m_q05": [-0.09, 0.10],
            "m_q25": [-0.02, -0.30],
            "m_q50": [0.015, 0.00],
            "m_q75": [0.045, -0.25],
            "m_q95": [0.10, 0.40],
            "y_q05": [-0.30, 0.5],
            "y_q25": [-0.05, -0.5],
            "y_q50": [0.05, 0.0],
            "y_q75": [0.20, -0.2],
            "y_q95": [0.50, 1.0],
            "converged": [True, True],
        }
    )


class TestAssembleForecasts:
    def test_output_has_expected_columns_and_row_count(self):
        out = assemble_forecasts(_final_pred(), _garch_df(), offsets={"90": 0.01, "50": 0.005}, as_of="2026-08-26")

        expected_cols = {
            "ticker",
            "as_of",
            "q05",
            "q25",
            "q50",
            "q75",
            "q95",
            "raw_q05",
            "raw_q25",
            "raw_q75",
            "raw_q95",
            "ens_q05",
            "ens_q25",
            "ens_q50",
            "ens_q75",
            "ens_q95",
            "y_q05",
            "y_q25",
            "y_q50",
            "y_q75",
            "y_q95",
            "ann_vol",
            "garch_converged",
        }
        assert expected_cols.issubset(out.columns)
        assert len(out) == 3
        assert set(out["ticker"]) == {"AAA", "BBB", "CCC"}
        assert (out["as_of"] == "2026-08-26").all()

    def test_monthly_display_quantiles_are_calibrated_values(self):
        out = assemble_forecasts(_final_pred(), _garch_df(), offsets={}, as_of="2026-08-26").set_index("ticker")
        row = out.loc["AAA"]
        assert row["q05"] == pytest.approx(-0.11)
        assert row["q25"] == pytest.approx(-0.035)
        assert row["q50"] == pytest.approx(0.01)
        assert row["q75"] == pytest.approx(0.055)
        assert row["q95"] == pytest.approx(0.13)
        assert row["raw_q05"] == pytest.approx(-0.10)
        assert row["raw_q25"] == pytest.approx(-0.03)
        assert row["raw_q75"] == pytest.approx(0.05)
        assert row["raw_q95"] == pytest.approx(0.12)

    def test_ensemble_math_exact_with_garch(self):
        out = assemble_forecasts(_final_pred(), _garch_df(), offsets={}, as_of="2026-08-26").set_index("ticker")
        row = out.loc["AAA"]
        assert row["ens_q05"] == pytest.approx((-0.11 + -0.09) / 2.0)
        assert row["ens_q25"] == pytest.approx((-0.035 + -0.02) / 2.0)
        assert row["ens_q50"] == pytest.approx((0.01 + 0.015) / 2.0)
        assert row["ens_q75"] == pytest.approx((0.055 + 0.045) / 2.0)
        assert row["ens_q95"] == pytest.approx((0.13 + 0.10) / 2.0)
        assert row["ann_vol"] == pytest.approx(0.25)
        assert row["garch_converged"] == True  # noqa: E712

    def test_ensemble_falls_back_to_lightgbm_when_garch_missing(self):
        out = assemble_forecasts(_final_pred(), _garch_df(), offsets={}, as_of="2026-08-26").set_index("ticker")
        row = out.loc["BBB"]
        assert row["ens_q05"] == pytest.approx(-0.085)
        assert row["ens_q25"] == pytest.approx(-0.025)
        assert row["ens_q50"] == pytest.approx(0.02)
        assert row["ens_q75"] == pytest.approx(0.045)
        assert row["ens_q95"] == pytest.approx(0.095)
        assert row["garch_converged"] == False  # noqa: E712
        assert pd.isna(row["ann_vol"])
        for col in ["y_q05", "y_q25", "y_q50", "y_q75", "y_q95"]:
            assert pd.isna(row[col])

    def test_yearly_recentered_on_twelve_times_ens_median(self):
        out = assemble_forecasts(_final_pred(), _garch_df(), offsets={}, as_of="2026-08-26").set_index("ticker")
        row = out.loc["AAA"]
        ens_q50 = (0.01 + 0.015) / 2.0
        target = 12.0 * ens_q50
        assert row["y_q50"] == pytest.approx(target)
        offset = target - 0.05  # 0.05 == raw y_q50 for AAA
        assert row["y_q05"] == pytest.approx(-0.30 + offset)
        assert row["y_q25"] == pytest.approx(-0.05 + offset)
        assert row["y_q75"] == pytest.approx(0.20 + offset)
        assert row["y_q95"] == pytest.approx(0.50 + offset)

    def test_yearly_bands_wider_than_monthly(self):
        out = assemble_forecasts(_final_pred(), _garch_df(), offsets={}, as_of="2026-08-26").set_index("ticker")
        row = out.loc["AAA"]
        monthly_width = row["q95"] - row["q05"]
        yearly_width = row["y_q95"] - row["y_q05"]
        assert yearly_width > monthly_width

    def test_monotonicity_enforced_on_all_three_quantile_families(self):
        out = assemble_forecasts(_final_pred(), _garch_df(), offsets={}, as_of="2026-08-26").set_index("ticker")
        row = out.loc["CCC"]  # deliberately non-monotonic garch inputs

        q_vals = row[["q05", "q25", "q50", "q75", "q95"]].to_numpy(dtype=float)
        ens_vals = row[["ens_q05", "ens_q25", "ens_q50", "ens_q75", "ens_q95"]].to_numpy(dtype=float)
        y_vals = row[["y_q05", "y_q25", "y_q50", "y_q75", "y_q95"]].to_numpy(dtype=float)

        assert np.all(np.diff(q_vals) >= -1e-12)
        assert np.all(np.diff(ens_vals) >= -1e-12)
        assert np.all(np.diff(y_vals) >= -1e-12)

        # ens should be exactly the sorted raw ensemble means.
        raw_ens = np.sort(
            np.array(
                [
                    (-0.05 + 0.10) / 2.0,
                    (-0.02 + -0.30) / 2.0,
                    (0.00 + 0.00) / 2.0,
                    (0.02 + -0.25) / 2.0,
                    (0.05 + 0.40) / 2.0,
                ]
            )
        )
        np.testing.assert_allclose(ens_vals, raw_ens)


def _wf_metrics() -> dict:
    return {
        "pinball_q50": np.float64(0.0123),
        "n_folds": np.int64(6),
        "sharpe": np.float32(1.5),
        "converged_all": np.bool_(True),
    }


def _offsets() -> dict:
    return {"90": 0.012, "50": np.float64(0.006), "coverage_raw_90": 0.83, "coverage_cal_90": 0.90}


def _backtest_summary() -> dict:
    return {"cum_return": np.float64(0.42), "folds": [np.int64(0), np.int64(1)], "ok": np.bool_(False)}


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": pd.to_datetime(["2025-01-31", "2025-02-28"]),
            "q05_cal": [-0.1, -0.09],
            "q95_cal": [0.1, 0.09],
            "y_true": [0.01, -0.02],
        }
    )


def _forecasts() -> pd.DataFrame:
    return assemble_forecasts(_final_pred(), _garch_df(), offsets=_offsets(), as_of="2026-08-26")


class TestWriteAndLoadArtifacts:
    def test_round_trip(self, tmp_path):
        forecasts = _forecasts()
        history = _history()
        write_artifacts(
            tmp_path,
            forecasts=forecasts,
            wf_metrics=_wf_metrics(),
            offsets=_offsets(),
            backtest_summary=_backtest_summary(),
            history=history,
            failures=["ZZZ"],
            staleness={"prices": False, "macro": True},
        )

        for name in [
            "forecasts.parquet",
            "oos_history.parquet",
            "backtest.json",
            "diagnostics.json",
            "manifest.json",
        ]:
            assert (tmp_path / name).exists()

        loaded = load_artifacts(tmp_path)
        assert set(loaded.keys()) == {"forecasts", "oos_history", "diagnostics", "backtest", "manifest"}

        pd.testing.assert_frame_equal(
            loaded["forecasts"].reset_index(drop=True), forecasts.reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            loaded["oos_history"].reset_index(drop=True), history.reset_index(drop=True)
        )

    def test_manifest_fields(self, tmp_path):
        forecasts = _forecasts()
        history = _history()
        write_artifacts(
            tmp_path,
            forecasts=forecasts,
            wf_metrics=_wf_metrics(),
            offsets=_offsets(),
            backtest_summary=_backtest_summary(),
            history=history,
            failures=["ZZZ", "YYY"],
            staleness={"prices": False},
        )
        loaded = load_artifacts(tmp_path)
        manifest = loaded["manifest"]

        # generated_at must be a parseable ISO-8601 UTC timestamp.
        parsed = pd.Timestamp(manifest["generated_at"])
        assert parsed.tzinfo is not None

        assert manifest["n_forecasts"] == len(forecasts)
        assert manifest["n_history_rows"] == len(history)
        assert manifest["failed_tickers"] == ["ZZZ", "YYY"]
        assert manifest["stale_sources"] == {"prices": False}
        for pkg in ["pandas", "lightgbm", "arch", "stockpred"]:
            assert pkg in manifest["versions"]
            assert isinstance(manifest["versions"][pkg], str)

    def test_diagnostics_contains_wf_metrics_and_offsets(self, tmp_path):
        wf_metrics = _wf_metrics()
        offsets = _offsets()
        write_artifacts(
            tmp_path,
            forecasts=_forecasts(),
            wf_metrics=wf_metrics,
            offsets=offsets,
            backtest_summary=_backtest_summary(),
            history=_history(),
            failures=[],
            staleness={},
        )
        loaded = load_artifacts(tmp_path)
        diag = loaded["diagnostics"]
        assert diag["wf_metrics"]["n_folds"] == 6
        assert diag["wf_metrics"]["pinball_q50"] == pytest.approx(0.0123)
        assert diag["offsets"]["90"] == pytest.approx(0.012)
        assert diag["offsets"]["coverage_cal_90"] == pytest.approx(0.90)

    def test_backtest_json_round_trips_numpy_types(self, tmp_path):
        write_artifacts(
            tmp_path,
            forecasts=_forecasts(),
            wf_metrics=_wf_metrics(),
            offsets=_offsets(),
            backtest_summary=_backtest_summary(),
            history=_history(),
            failures=[],
            staleness={},
        )
        loaded = load_artifacts(tmp_path)
        bt = loaded["backtest"]
        assert bt["cum_return"] == pytest.approx(0.42)
        assert bt["folds"] == [0, 1]
        assert bt["ok"] is False

    def test_mkdir_p_creates_nested_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "artifacts"
        write_artifacts(
            nested,
            forecasts=_forecasts(),
            wf_metrics=_wf_metrics(),
            offsets=_offsets(),
            backtest_summary=_backtest_summary(),
            history=_history(),
            failures=[],
            staleness={},
        )
        assert (nested / "manifest.json").exists()


class TestLoadArtifactsMissingAndCorrupt:
    def test_missing_directory_returns_all_none(self, tmp_path):
        missing_dir = tmp_path / "does_not_exist"
        loaded = load_artifacts(missing_dir)
        assert loaded == {
            "forecasts": None,
            "oos_history": None,
            "diagnostics": None,
            "backtest": None,
            "manifest": None,
        }

    def test_missing_individual_files_map_to_none(self, tmp_path):
        write_artifacts(
            tmp_path,
            forecasts=_forecasts(),
            wf_metrics=_wf_metrics(),
            offsets=_offsets(),
            backtest_summary=_backtest_summary(),
            history=_history(),
            failures=[],
            staleness={},
        )
        (tmp_path / "backtest.json").unlink()
        (tmp_path / "oos_history.parquet").unlink()

        loaded = load_artifacts(tmp_path)
        assert loaded["backtest"] is None
        assert loaded["oos_history"] is None
        assert loaded["forecasts"] is not None
        assert loaded["manifest"] is not None

    def test_corrupt_json_raises(self, tmp_path):
        write_artifacts(
            tmp_path,
            forecasts=_forecasts(),
            wf_metrics=_wf_metrics(),
            offsets=_offsets(),
            backtest_summary=_backtest_summary(),
            history=_history(),
            failures=[],
            staleness={},
        )
        (tmp_path / "manifest.json").write_text("{not valid json")

        with pytest.raises(json.JSONDecodeError):
            load_artifacts(tmp_path)

    def test_corrupt_parquet_raises(self, tmp_path):
        write_artifacts(
            tmp_path,
            forecasts=_forecasts(),
            wf_metrics=_wf_metrics(),
            offsets=_offsets(),
            backtest_summary=_backtest_summary(),
            history=_history(),
            failures=[],
            staleness={},
        )
        (tmp_path / "forecasts.parquet").write_bytes(b"not a parquet file")

        with pytest.raises(Exception):  # noqa: B017 - pyarrow's own exception type
            load_artifacts(tmp_path)

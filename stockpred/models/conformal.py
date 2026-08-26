"""Split conformalized quantile regression (CQR) calibration (Task 7).

``run_walk_forward`` (see :mod:`stockpred.models.workhorse`) produces, per
fold, a genuinely held-out ``cal`` set of quantile predictions (from a model
fit only on the fold's earlier ``fit`` months) alongside the fold's ``oos``
test predictions (from a model fit on ``fit`` + ``cal`` months). Because the
underlying quantile regressor has no coverage guarantee, its predicted
``[q05, q95]`` and ``[q25, q75]`` intervals can be systematically too narrow
(under-covering) or too wide (over-covering).

Split CQR (Romano, Patterson & Candes 2019) fixes this by computing a single
scalar offset per interval from the calibration set's conformity scores and
adding it to both interval bounds. This keeps the finite-sample marginal
coverage guarantee (under exchangeability) without touching the model
itself.

``calibrate_from_wf`` computes those offsets **per fold** -- fold ``k``'s
offsets come from fold ``k``'s own calibration rows, which strictly precede
fold ``k``'s test months -- so no reported interval is ever a function of
residuals observed after the month it is applied to. See that function's
docstring for why the older pooled-across-folds version was a look-ahead
leak.

All three functions here operate on returns as decimal fractions, matching
the rest of the codebase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from stockpred.models.workhorse import WalkForwardResult

_MIN_CAL_N = 20


def cqr_offsets(cal_lo, cal_hi, cal_y, alpha: float) -> float:
    """Split-CQR conformity offset for one interval ``[cal_lo, cal_hi]``.

    Conformity score for calibration point ``i``:
        ``E_i = max(cal_lo_i - y_i, y_i - cal_hi_i)``

    (positive when ``y_i`` falls outside the predicted interval on either
    side, negative -- interval too wide -- when comfortably inside).

    The returned offset is the finite-sample-corrected empirical quantile of
    ``E`` at level ``min(1, ceil((n+1)(1-alpha)) / n)`` (Romano et al. 2019,
    "Conformalized Quantile Regression"): with ``n`` exchangeable
    calibration points, this is the smallest quantile level for which
    ``[lo - offset, hi + offset]`` covers a fresh exchangeable test point
    with probability >= ``1 - alpha``, using ``np.quantile(..., method=
    "higher")`` so the quantile lands exactly on one of the ``n`` observed
    scores (never interpolated), which is what makes the finite-sample
    guarantee exact rather than asymptotic.

    Adding this offset to both interval bounds (widening if positive,
    shrinking if negative) is what ``apply_cqr`` does.

    Raises ``ValueError`` if ``n < 20`` -- below that the empirical quantile
    is too noisy to be a meaningful calibration target.
    """
    cal_lo = np.asarray(cal_lo, dtype=float)
    cal_hi = np.asarray(cal_hi, dtype=float)
    cal_y = np.asarray(cal_y, dtype=float)
    n = len(cal_y)
    if n < _MIN_CAL_N:
        raise ValueError(
            f"cqr_offsets requires a meaningful calibration set (n >= {_MIN_CAL_N}), got n={n}"
        )

    scores = np.maximum(cal_lo - cal_y, cal_y - cal_hi)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def apply_cqr(pred: pd.DataFrame, offsets: dict) -> pd.DataFrame:
    """Apply precomputed CQR offsets to a predictions frame.

    Adds ``q05_cal, q25_cal, q75_cal, q95_cal`` columns:
        ``q05_cal = q05 - offsets["90"]``, ``q95_cal = q95 + offsets["90"]``
        ``q25_cal = q25 - offsets["50"]``, ``q75_cal = q75 + offsets["50"]``

    Offsets may be negative (over-wide intervals shrink) -- that's valid
    CQR. Because the two offsets are applied independently to the 90% and
    50% intervals, a large enough offset difference could in principle push
    ``q25_cal`` below ``q05_cal`` or above ``q75_cal_before_widening``, etc.
    To guarantee a proper nested interval, the five values
    ``[q05_cal, q25_cal, q50, q75_cal, q95_cal]`` are sorted per row and
    reassigned: positions 0, 1, 3, 4 become the four ``*_cal`` columns.
    Position 2 (the sorted median) is discarded -- the ``q50`` column is
    left untouched at its original value, per the calibration contract (only
    the tails are calibrated). A crossing q50 relative to the sorted middle
    is theoretically possible but harmless: the sort still guarantees
    ``q05_cal <= q25_cal <= q75_cal <= q95_cal``, which is the invariant
    downstream consumers need.

    The original ``q05..q95`` columns are preserved untouched.
    """
    out = pred.copy()
    q05_cal = pred["q05"].to_numpy(dtype=float) - offsets["90"]
    q95_cal = pred["q95"].to_numpy(dtype=float) + offsets["90"]
    q25_cal = pred["q25"].to_numpy(dtype=float) - offsets["50"]
    q75_cal = pred["q75"].to_numpy(dtype=float) + offsets["50"]
    q50 = pred["q50"].to_numpy(dtype=float)

    stacked = np.column_stack([q05_cal, q25_cal, q50, q75_cal, q95_cal])
    stacked.sort(axis=1)

    out["q05_cal"] = stacked[:, 0]
    out["q25_cal"] = stacked[:, 1]
    # stacked[:, 2] (sorted median) intentionally discarded; q50 unchanged.
    out["q75_cal"] = stacked[:, 3]
    out["q95_cal"] = stacked[:, 4]
    return out


def calibrate_from_wf(wf: "WalkForwardResult") -> tuple[dict, pd.DataFrame]:
    """Calibrate a walk-forward result's ``oos`` predictions **per fold**,
    each fold using only its own ``cal`` rows.

    Why per fold (and why pooling would be a look-ahead leak)
    ----------------------------------------------------------
    An earlier version of this function pooled every fold's ``cal`` rows
    into one calibration set and applied the resulting scalar offsets to
    every fold's ``oos`` rows. The argument given for that being leak-free
    was *model disjointness*: no ``cal`` row was ever used to fit the model
    that produced the ``oos`` predictions. That argument is wrong, because
    model disjointness is not temporal disjointness. The pooled ``cal`` set
    spans the whole sample, so the offset applied to fold 0's 2018 test
    months was computed partly from calibration residuals observed in 2023-
    2025 -- information that did not exist in 2018. The interval widths (and
    therefore the ``q05_cal`` gate that drives the backtest's position
    selection) were then a function of the future. Measured on the real
    run, that inflated the reported backtest materially.

    So: fold ``k``'s offsets come from fold ``k``'s own ``cal`` rows only,
    which are the trailing ``CAL_MONTHS`` of fold ``k``'s *training* block
    and therefore strictly precede fold ``k``'s test block (the walk-forward
    harness embargoes the gap between them -- see
    :func:`stockpred.validation.purged_walk_forward`). Every fold's
    calibration set is far larger than ``_MIN_CAL_N`` in practice (450+ rows
    on the real panel), so nothing is lost in statistical precision.

    Returns ``(offsets, oos_calibrated)``.

    ``offsets`` is::

        {"90": float,            # POOLED offset -- diagnostic display only
         "50": float,            # POOLED offset -- diagnostic display only
         "per_fold": {"0": {"90": float, "50": float}, "1": {...}, ...},
         "coverage_raw_90": float,
         "coverage_cal_90": float}

    The top-level ``"90"``/``"50"`` values are the *pooled* offsets, kept
    only as a whole-sample summary statistic for ``diagnostics.json`` and
    the dashboard's "conformal calibration offsets" line. **Nothing in the
    pipeline calibrates anything with them**: the reported ``oos`` intervals
    here use ``per_fold``, and the shipped production forecast computes its
    own offsets from its own trailing calibration block (see
    :func:`stockpred.pipeline._production_forecast`). They are a summary,
    not an input.

    ``per_fold`` is keyed by the fold id as a string, so the dict round-
    trips through JSON unchanged.

    ``coverage_raw_90``/``coverage_cal_90`` are the empirical fraction of
    ``oos`` rows whose ``y_true`` falls inside ``[q05, q95]`` and
    ``[q05_cal, q95_cal]`` respectively -- both computed on the
    per-fold-calibrated frame, so the calibrated figure is an honest
    out-of-sample coverage number.

    ``oos_calibrated`` is ``wf.oos`` with the four ``*_cal`` columns added
    (see ``apply_cqr``), in ``wf.oos``'s original row order.
    """
    cal = wf.cal
    oos = wf.oos

    # Pooled offsets: reported as a summary statistic only (see docstring).
    pooled: dict = {
        "90": cqr_offsets(cal["q05"], cal["q95"], cal["y_true"], alpha=0.10),
        "50": cqr_offsets(cal["q25"], cal["q75"], cal["y_true"], alpha=0.50),
    }

    per_fold: dict[str, dict[str, float]] = {}
    calibrated_parts: list[pd.DataFrame] = []

    for fold_id in sorted(oos["fold"].unique()):
        fold_cal = cal[cal["fold"] == fold_id]
        fold_oos = oos[oos["fold"] == fold_id]
        if fold_cal.empty:
            raise ValueError(
                f"calibrate_from_wf: fold {fold_id} has oos rows but no cal "
                "rows -- cannot compute an honest per-fold offset for it"
            )
        fold_offsets = {
            "90": cqr_offsets(fold_cal["q05"], fold_cal["q95"], fold_cal["y_true"], alpha=0.10),
            "50": cqr_offsets(fold_cal["q25"], fold_cal["q75"], fold_cal["y_true"], alpha=0.50),
        }
        per_fold[str(fold_id)] = fold_offsets
        calibrated_parts.append(apply_cqr(fold_oos, fold_offsets))

    # Reassemble in the input's original row order (each part kept wf.oos's
    # own index labels, which are unique).
    oos_calibrated = pd.concat(calibrated_parts).reindex(oos.index)

    y_true = oos_calibrated["y_true"]
    offsets: dict = dict(pooled)
    offsets["per_fold"] = per_fold
    offsets["coverage_raw_90"] = float(
        ((oos_calibrated["q05"] <= y_true) & (y_true <= oos_calibrated["q95"])).mean()
    )
    offsets["coverage_cal_90"] = float(
        ((oos_calibrated["q05_cal"] <= y_true) & (y_true <= oos_calibrated["q95_cal"])).mean()
    )

    return offsets, oos_calibrated

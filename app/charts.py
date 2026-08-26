"""Pure Plotly figure builders for the StockPred Streamlit dashboard.

No Streamlit (or ``stockpred``) imports here on purpose: every function is a
plain ``DataFrame``/``dict`` -> ``go.Figure`` transform, so it can be
exercised directly in tests without spinning up an app. Colors are chosen to
read cleanly on both light and dark Streamlit themes -- the default Plotly
template with transparent paper/plot backgrounds, rather than
``"plotly_dark"`` (which looks wrong under a light theme).
"""

from __future__ import annotations

from typing import Any, Sequence

import plotly.graph_objects as go

_BAND_90 = "rgba(31, 119, 180, 0.18)"
_BAND_50 = "rgba(31, 119, 180, 0.38)"
_MEDIAN = "#1f77b4"
_HIST = "#7f7f7f"
_BENCH = "#d62728"
_GOOD = "#2ca02c"
_TARGET = "#ff7f0e"


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def fan_chart(
    hist_dates: Sequence[Any],
    hist_index: Sequence[float],
    fan_date: Any,
    quantiles: dict[str, float],
    title: str = "Price index (base 100) + 1-month fan",
) -> go.Figure:
    """Historical cumulative return index (base 100) plus a one-step fan.

    ``hist_dates``/``hist_index`` are the trailing history (e.g. last 24
    months), already cumulated to an index starting at 100. ``quantiles``
    holds the *monthly return* quantiles (decimal fractions, keys
    ``q05,q25,q50,q75,q95``) for the next step; the fan's y-values are
    ``last_index * (1 + q)``, drawn as a single segment from the last
    historical point to ``fan_date``. Bands: 5-95% light fill, 25-75%
    darker fill, median as a dashed line.
    """
    fig = go.Figure()

    hist_dates = list(hist_dates)
    hist_index = list(hist_index)

    if hist_dates:
        fig.add_trace(
            go.Scatter(
                x=hist_dates,
                y=hist_index,
                mode="lines",
                name="Historical (indexed to 100)",
                line=dict(color=_HIST, width=2),
            )
        )
        last_date = hist_dates[-1]
        last_index = hist_index[-1]
    else:
        last_date = fan_date
        last_index = 100.0

    x_fan = [last_date, fan_date]

    def _band(lo_key: str, hi_key: str, color: str, name: str) -> None:
        y_lo = [last_index, last_index * (1 + quantiles[lo_key])]
        y_hi = [last_index, last_index * (1 + quantiles[hi_key])]
        fig.add_trace(
            go.Scatter(
                x=x_fan, y=y_lo, mode="lines", line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_fan, y=y_hi, mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=color, name=name,
            )
        )

    _band("q05", "q95", _BAND_90, "5-95% band")
    _band("q25", "q75", _BAND_50, "25-75% band")

    y_med = [last_index, last_index * (1 + quantiles["q50"])]
    fig.add_trace(
        go.Scatter(
            x=x_fan, y=y_med, mode="lines+markers", name="Median",
            line=dict(color=_MEDIAN, width=2, dash="dash"),
        )
    )

    return _base_layout(fig, title)


def humble_band_chart(
    y_quantiles: dict[str, float],
    title: str = "12-month scenario range (wide, humble -- not a prediction)",
) -> go.Figure:
    """Horizontal interval plot of the yearly (12-month) quantile band."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[y_quantiles["q95"] - y_quantiles["q05"]],
            y=["12-month"],
            base=[y_quantiles["q05"]],
            orientation="h",
            marker=dict(color=_BAND_90),
            name="5-95%",
        )
    )
    fig.add_trace(
        go.Bar(
            x=[y_quantiles["q75"] - y_quantiles["q25"]],
            y=["12-month"],
            base=[y_quantiles["q25"]],
            orientation="h",
            marker=dict(color=_BAND_50),
            name="25-75%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[y_quantiles["q50"]],
            y=["12-month"],
            mode="markers",
            marker=dict(symbol="line-ns", size=40, color=_MEDIAN, line=dict(width=3, color=_MEDIAN)),
            name="Median",
        )
    )
    fig.update_layout(barmode="overlay", xaxis=dict(tickformat=".1%"))
    return _base_layout(fig, title)


def r2_bar(
    fold_r2: dict[Any, float],
    pooled_r2: float | None = None,
    title: str = "OOS R² by fold (vs historical-mean benchmark)",
) -> go.Figure:
    """Bar chart of per-fold ``r2_oos_median``, with a benchmark line at 0
    and an optional pooled-R² reference line.
    """
    folds = sorted(fold_r2.keys(), key=lambda f: int(f))
    values = [fold_r2[f] for f in folds]
    colors = [_GOOD if v is not None and v >= 0 else _BENCH for v in values]

    fig = go.Figure(
        go.Bar(
            x=[f"Fold {f}" for f in folds],
            y=values,
            marker=dict(color=colors),
            name="R² (OOS median vs historical-mean benchmark)",
        )
    )
    fig.add_hline(y=0, line=dict(color=_TARGET, dash="dot"), annotation_text="benchmark (R²=0)")
    if pooled_r2 is not None:
        fig.add_hline(
            y=pooled_r2, line=dict(color=_MEDIAN, dash="dash"),
            annotation_text=f"pooled R²={pooled_r2:.3f}",
        )
    return _base_layout(fig, title)


def coverage_chart(
    coverage_raw: float,
    coverage_cal: float,
    target: float = 0.90,
    title: str = "90% interval coverage: raw vs calibrated",
) -> go.Figure:
    """Two bullet-style horizontal bars (raw vs CQR-calibrated coverage)
    against a target line.
    """
    fig = go.Figure(
        go.Bar(
            y=["Raw (uncalibrated)", "Calibrated (CQR)"],
            x=[coverage_raw, coverage_cal],
            orientation="h",
            marker=dict(color=[_HIST, _MEDIAN]),
        )
    )
    fig.add_vline(x=target, line=dict(color=_TARGET, dash="dash"), annotation_text=f"target {target:.0%}")
    fig.update_layout(xaxis=dict(range=[0, 1], tickformat=".0%"))
    return _base_layout(fig, title)


def feature_importance_bar(
    records: list[dict[str, Any]],
    top_n: int = 20,
    title: str = "Feature importance (mean LightGBM gain, q50 model, across folds)",
) -> go.Figure:
    """Horizontal bar chart of walk-forward feature importance.

    ``records`` is ``diagnostics.json``'s ``feature_importance`` list --
    ``[{"feature": str, "importance": float}, ...]``. The chart shows the
    ``top_n`` most important features, largest at the top (Plotly draws a
    horizontal bar's first category at the bottom, so the list is reversed
    before plotting).
    """
    ranked = sorted(records, key=lambda r: float(r["importance"]), reverse=True)[:top_n]
    ranked.reverse()  # so the largest ends up at the TOP of the drawn axis

    fig = go.Figure(
        go.Bar(
            x=[float(r["importance"]) for r in ranked],
            y=[str(r["feature"]) for r in ranked],
            orientation="h",
            marker=dict(color=_MEDIAN),
            name="mean gain",
        )
    )
    fig.update_layout(
        xaxis_title="mean gain", showlegend=False,
        height=max(280, 22 * len(ranked) + 120),
    )
    return _base_layout(fig, title)


def equity_curve(
    monthly_returns: list[dict[str, Any]],
    benchmark_total_return: float | None = None,
    initial_capital: float = 10_000.0,
    title: str = "Strategy equity vs buy-hold total-return endpoint",
) -> go.Figure:
    """Strategy equity curve (net-of-cost compounding) plus a horizontal
    marker for the buy-hold benchmark's total-return endpoint.

    Per-month benchmark equity isn't available in the backtest artifact
    (only its compounded total return is) -- the horizontal line marks
    where a buy-and-hold portfolio would have *ended up*, not a real path,
    and is labeled as such.
    """
    dates = [m["date"] for m in monthly_returns]
    capital = initial_capital
    equity: list[float] = []
    for m in monthly_returns:
        capital *= 1.0 + m["ret"]
        equity.append(capital)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates, y=equity, mode="lines", name="Strategy equity (net of costs)",
            line=dict(color=_MEDIAN, width=2),
        )
    )

    if benchmark_total_return is not None and dates:
        bench_final = initial_capital * (1.0 + benchmark_total_return)
        fig.add_trace(
            go.Scatter(
                x=[dates[0], dates[-1]], y=[bench_final, bench_final],
                mode="lines", name="Buy-hold total-return endpoint (not a path)",
                line=dict(color=_BENCH, dash="dot"),
            )
        )

    fig.update_layout(yaxis_title="Capital (EUR)")
    return _base_layout(fig, title)

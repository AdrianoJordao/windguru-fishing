"""
    Interactive Plotly chart: one panel per variable plus tides, with the
    good/near/bad hours painted behind every panel.
"""

from __future__ import annotations

from datetime import tzinfo
from typing import Dict, Optional, Tuple

import pandas as pd
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import DEFAULT_RANGES, LOCAL_TZ, VARIABLES
from i18n import DEFAULT_LANG, fmt_day, t
from windguru_scraper import PlotData

STATUS_COLORS = {"green": "#22c55e", "yellow": "#facc15", "red": "#f87171"}
STATUS_OPACITY = 0.18
RANGE_COLOR = "rgba(59, 130, 246, 0.12)"
MEAN_COLOR = "#111827"
TIDE_COLOR = "#0e7490"
ROW_HEIGHT_PX = 190


def to_local_naive(idx, tz: tzinfo) -> pd.DatetimeIndex:
    """ UTC timestamps -> naive local time (Plotly has no timezone support) """
    return pd.DatetimeIndex(idx).tz_convert(tz).tz_localize(None)


def build_figure(plot_data: PlotData,
                 ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                 tide_curve: Optional[pd.Series] = None,
                 tide_extremes: Optional[pd.DataFrame] = None,
                 title: str = "",
                 tz: tzinfo = LOCAL_TZ,
                 lang: str = DEFAULT_LANG) -> go.Figure:
    ranges = ranges or DEFAULT_RANGES

    def _local(idx) -> pd.DatetimeIndex:
        return to_local_naive(idx, tz)

    titles = [f"{t(f'var.{v.col}.label', lang)} ({v.unit})" for v in VARIABLES] + [t("chart.tide_panel", lang)]
    mean_name, mean_short = t("chart.mean", lang), t("chart.mean_short", lang)
    n_rows = len(titles)
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                        vertical_spacing=0.025, subplot_titles=titles)
    # Panel titles on the left, so the day labels above the first panel stay free
    fig.update_annotations(x=0, xanchor="left", font=dict(size=13))

    # Consistent colour per model across all panels
    palette = plotly.colors.qualitative.Plotly + plotly.colors.qualitative.Safe
    models = list(dict.fromkeys(plot_data.interp["model"])) if not plot_data.interp.empty else []
    model_color = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    shown_in_legend: set = set()

    x_mean = _local(plot_data.means.index) if len(plot_data.means) else []

    for row, var in enumerate(VARIABLES, start=1):
        if var.col in ranges:
            lo, hi = ranges[var.col]
            fig.add_hrect(y0=lo, y1=hi, fillcolor=RANGE_COLOR, line_width=0,
                          layer="below", row=row, col=1)

        for model in models:
            g = plot_data.interp[plot_data.interp["model"] == model]
            if not g[var.col].notna().any():
                continue
            fig.add_trace(go.Scatter(
                x=_local(g["datetime"]), y=g[var.col], name=model,
                legendgroup=model, showlegend=model not in shown_in_legend,
                mode="lines+markers", marker=dict(size=3), line=dict(width=1, color=model_color[model]),
                hovertemplate=f"{model}: %{{y:.1f}} {var.unit}<extra></extra>",
            ), row=row, col=1)
            shown_in_legend.add(model)

        if var.col in plot_data.means and plot_data.means[var.col].notna().any():
            fig.add_trace(go.Scatter(
                x=x_mean, y=plot_data.means[var.col], name=mean_name,
                legendgroup="mean", showlegend="mean" not in shown_in_legend,
                mode="lines", line=dict(width=2.2, color=MEAN_COLOR),
                hovertemplate=f"<b>{mean_short}: %{{y:.1f}} {var.unit}</b><extra></extra>",
            ), row=row, col=1)
            shown_in_legend.add("mean")

    # ---------------- Tides ----------------
    tide_row = n_rows
    has_tides = tide_curve is not None and not tide_curve.empty
    if has_tides:
        curve = tide_curve
        if plot_data.start is not None and plot_data.end is not None:
            curve = curve[(curve.index >= plot_data.start) & (curve.index <= plot_data.end)]
        fig.add_trace(go.Scatter(
            x=_local(curve.index), y=curve.values, name=t("chart.tide", lang),
            mode="lines", line=dict(width=2, color=TIDE_COLOR), fill="tozeroy",
            fillcolor="rgba(14, 116, 144, 0.10)",
            hovertemplate=f"{t('chart.tide', lang)}: %{{y:.0f}} cm<extra></extra>",
        ), row=tide_row, col=1)

        if tide_extremes is not None and not tide_extremes.empty:
            ext = tide_extremes
            if plot_data.start is not None and plot_data.end is not None:
                ext = ext[(ext["time"] >= plot_data.start) & (ext["time"] <= plot_data.end)]
            for kind, symbol, pos in (("High", "triangle-up", "top center"),
                                      ("Low", "triangle-down", "bottom center")):
                e = ext[ext["type"] == kind]
                if e.empty:
                    continue
                x = _local(e["time"])
                kind_name = t(f"tide.{kind}", lang)
                fig.add_trace(go.Scatter(
                    x=x, y=e["height_cm"], name=kind_name, showlegend=False,
                    mode="markers+text", marker=dict(symbol=symbol, size=9, color=TIDE_COLOR),
                    text=[f"{t:%H:%M}<br>{h:+.0f}" for t, h in zip(x, e["height_cm"])],
                    textposition=pos, textfont=dict(size=10),
                    hovertemplate=f"{kind_name} %{{x|%H:%M}}: %{{y:+.0f}} cm<extra></extra>",
                ), row=tide_row, col=1)
            lo, hi = float(curve.min()), float(curve.max())
            pad = max(25.0, 0.35 * (hi - lo))
            fig.update_yaxes(range=[lo - pad, hi + pad], row=tide_row, col=1)
    else:
        fig.add_annotation(text=t("chart.no_tide", lang), showarrow=False, font=dict(color="gray"),
                           xref=f"x{tide_row} domain", yref=f"y{tide_row} domain", x=0.5, y=0.5)

    # ---------------- Good / near / bad hours behind every panel ----------------
    for t0, t1, status in plot_data.spans:
        fig.add_vrect(x0=_local([t0])[0], x1=_local([t1])[0],
                      fillcolor=STATUS_COLORS.get(status, STATUS_COLORS["red"]),
                      opacity=STATUS_OPACITY, line_width=0, layer="below", row="all", col=1)

    # Day separators and day labels on top
    if plot_data.start is not None and plot_data.end is not None:
        first = _local([plot_data.start])[0].normalize()
        last = _local([plot_data.end])[0]
        for day in pd.date_range(first, last, freq="D"):
            if day > first:
                fig.add_vline(x=day, line_width=1, line_dash="dot", line_color="gray",
                              opacity=0.6, row="all", col=1)
            fig.add_annotation(x=day + pd.Timedelta(hours=12), y=1.0, xref="x", yref="paper",
                               yanchor="bottom", yshift=22, text=f"<b>{fmt_day(day, lang)}</b>", showarrow=False,
                               font=dict(size=12))
        fig.update_xaxes(range=[first, _local([plot_data.end])[0]])

    fig.update_xaxes(dtick=3 * 3600 * 1000, tickformat="%Hh", showgrid=True,
                     gridcolor="rgba(128,128,128,0.15)")
    fig.update_xaxes(showticklabels=True, row=n_rows, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)", zeroline=False)

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", y=1, yref="container", yanchor="top",
                   pad=dict(t=12), font=dict(size=18)),
        height=ROW_HEIGHT_PX * n_rows + 120,
        margin=dict(l=50, r=20, t=110, b=90),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.03, x=0, xanchor="left"),
    )
    return fig

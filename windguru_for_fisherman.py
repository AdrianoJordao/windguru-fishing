"""
    A small app to find good days for fishing based on data provided by Windguru
    Created on:      23-Oct-2025
    Original author: Adriano Jordão
"""

from __future__ import annotations

import logging
import re
from itertools import groupby
from typing import Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config import (DEFAULT_SPOT_NAME, DEFAULT_SPOT_URL, KNOWN_SPOTS, LOCAL_TZ,
                    VARIABLES)
from forecast_plotly import build_figure
from i18n import DEFAULT_LANG, LANGS, fmt_day, fmt_day_time, t
from tides import fetch_tide_curve, find_extremes
from windguru_scraper import (WindguruScraper, best_windows, daily_summary,
                              good_hours, parse_spot_id, prepare_plot_data)

logger = logging.getLogger(__name__)

# ------------------------------ Language ------------------------------
# Default is Portuguese; ?lang=en in the URL opens the app in English.
if "lang" not in st.session_state:
    q = st.query_params.get("lang")
    st.session_state.lang = q if q in LANGS else DEFAULT_LANG
st.session_state.setdefault("lang_choice", st.session_state.lang)


def _on_lang_change() -> None:
    choice = st.session_state.lang_choice
    if choice in LANGS:
        st.session_state.lang = choice
        st.query_params["lang"] = choice
    else:  # the toggle was clicked off: keep the current language selected
        st.session_state.lang_choice = st.session_state.lang


lang: str = st.session_state.lang

# Page configuration
st.set_page_config(
    page_title=t("page_title", lang),
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="auto"   # open on desktop, closed on phones
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 3rem !important; }
        h1 { margin-top: 0rem !important; margin-bottom: 0.5rem !important; }
        @media (min-width: 768px) {
            section[data-testid="stSidebar"] { width: 420px !important; }
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------ Cached data loaders ------------------------------

@st.cache_data(ttl=1800, show_spinner=False)
def load_forecast(spot_id: str, spot_url: str):
    """ Windguru forecast for a spot, cached for 30 min so we don't hammer Windguru """
    scraper = WindguruScraper(spot_url)
    df = scraper.fetch_data()
    return df, scraper.spot_name, scraper.lat, scraper.lon


@st.cache_data(ttl=3600, show_spinner=False)
def load_tides(lat: float, lon: float) -> Tuple[pd.Series, pd.DataFrame]:
    curve = fetch_tide_curve(lat, lon)
    return curve, find_extremes(curve)


def resolve_tide_coords(spot_id: str, lat_text: str, lon_text: str,
                        payload_lat: Optional[float], payload_lon: Optional[float]
                        ) -> Optional[Tuple[float, float, str]]:
    """ Manual input first, then Windguru's coordinates, then the known spots table """
    if lat_text.strip() and lon_text.strip():
        try:
            return float(lat_text.replace(",", ".")), float(lon_text.replace(",", ".")), "manual"
        except ValueError:
            st.warning(t("warn_bad_coords", lang))
    if payload_lat is not None and payload_lon is not None:
        return payload_lat, payload_lon, "windguru"
    if spot_id in KNOWN_SPOTS:
        _, lat, lon = KNOWN_SPOTS[spot_id]
        return lat, lon, "builtin"
    return None


def var_label(col: str, unit: Optional[str] = None) -> str:
    label = t(f"var.{col}.label", lang)
    return f"{label} ({unit})" if unit else label


MOBILE_UA = re.compile(r"Mobi|Android|iPhone|iPod|Windows Phone", re.IGNORECASE)


def is_phone() -> bool:
    """ Best guess from the browser's User-Agent; the sidebar toggle can override it """
    try:
        ua = st.context.headers.get("User-Agent", "") or ""
    except Exception:
        ua = ""
    return bool(MOBILE_UA.search(ua))


def _week_rows(plot_data, tide_extremes) -> list:
    """ One dict per day: day, good hours, best stretch, high and low tide times """
    summary = daily_summary(plot_data)
    highs: dict = {}
    lows: dict = {}
    if tide_extremes is not None and not tide_extremes.empty:
        for ts, kind in zip(tide_extremes["time"].dt.tz_convert(LOCAL_TZ), tide_extremes["type"]):
            (highs if kind == "High" else lows).setdefault(ts.date(), []).append(f"{ts:%H:%M}")

    def stretch(a, b) -> str:
        if pd.isna(a):
            return "—"
        if b - a >= pd.Timedelta(days=1):
            return t("all_day", lang)
        end = "24:00" if b.date() > a.date() and b.hour == 0 and b.minute == 0 else f"{b:%H:%M}"
        return f"{a:%H:%M}–{end}"

    return [{"day": d, "good": int(g), "best": stretch(a, b),
             "highs": highs.get(d, []), "lows": lows.get(d, [])}
            for d, g, a, b in zip(summary["date"], summary["green_hours"],
                                  summary["best_start"], summary["best_end"])]


def week_table(rows: list) -> pd.DataFrame:
    """ Desktop: a normal table with one column per item """
    return pd.DataFrame({
        t("col.day", lang): [fmt_day(r["day"], lang) for r in rows],
        t("col.good_hours", lang): [r["good"] for r in rows],
        t("col.best", lang): [r["best"] for r in rows],
        t("col.high", lang): [" · ".join(r["highs"]) or "—" for r in rows],
        t("col.low", lang): [" · ".join(r["lows"]) or "—" for r in rows],
    })


def week_html(rows: list) -> str:
    """ Phone: a narrow table (4 columns, tides stacked) that fits a ~360 px screen """
    body = []
    for r in rows:
        tides = (f'<span class="wk-hi">▲ {" ".join(r["highs"]) or "—"}</span><br>'
                 f'<span class="wk-lo">▼ {" ".join(r["lows"]) or "—"}</span>')
        good = f'<b>{r["good"]}</b>' if r["good"] else '<span class="wk-dim">0</span>'
        body.append(f'<tr><td>{fmt_day(r["day"], lang)}</td><td class="wk-num">{good}</td>'
                    f'<td>{r["best"]}</td><td class="wk-tide">{tides}</td></tr>')
    return f"""
<style>
  table.wk {{ width: 100%; border-collapse: collapse; font-size: 0.76rem; line-height: 1.3; }}
  table.wk th, table.wk td {{ padding: 0.35rem 0.2rem; text-align: left; vertical-align: middle;
                             border-bottom: 1px solid rgba(128,128,128,0.25); white-space: nowrap; }}
  table.wk th {{ font-weight: 600; opacity: 0.75; }}
  table.wk .wk-num {{ text-align: center; }}
  table.wk .wk-dim {{ opacity: 0.45; }}
  table.wk .wk-tide {{ font-variant-numeric: tabular-nums; }}
  table.wk .wk-hi {{ color: #0e7490; }}
  table.wk .wk-lo {{ color: #64748b; }}
</style>
<table class="wk">
  <thead><tr><th>{t("col.day", lang)}</th><th class="wk-num">{t("col.good_short", lang)}</th>
  <th>{t("col.best_short", lang)}</th><th>{t("col.tides", lang)}</th></tr></thead>
  <tbody>{"".join(body)}</tbody>
</table>
"""


def close_sidebar() -> None:
    """
    Collapse the sidebar by clicking Streamlit's own close button. On phones the
    sidebar covers the whole screen, so close it once the forecast is requested.
    The counter makes each call a new iframe, so the script runs every time.
    """
    st.session_state["_close_sidebar_n"] = st.session_state.get("_close_sidebar_n", 0) + 1
    components.html(f"""
        <script>
        // call {st.session_state["_close_sidebar_n"]}
        // Only click "close" if the sidebar is actually on screen; on some
        // Streamlit versions the same button would open a closed sidebar.
        const doc = window.parent.document;
        const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
        const r = sidebar ? sidebar.getBoundingClientRect() : null;
        const btn = doc.querySelector('[data-testid="stSidebarCollapseButton"] button');
        if (r && r.width > 0 && r.right > 0 && btn) btn.click();
        </script>
    """, height=0)


# ------------------------------ Session state defaults ------------------------------

for var in VARIABLES:
    st.session_state.setdefault(f"{var.col}_min", var.default[0])
    st.session_state.setdefault(f"{var.col}_max", var.default[1])
st.session_state.setdefault("spot_url", DEFAULT_SPOT_URL)
st.session_state.setdefault("spot_name", DEFAULT_SPOT_NAME)
st.session_state.setdefault("tide_lat", "")
st.session_state.setdefault("tide_lon", "")
st.session_state.setdefault("has_results", False)
st.session_state.setdefault("compact", is_phone())
st.session_state.setdefault("show_models", False)

# ------------------------------ Header: title + language toggle ------------------------------
head_left, head_right = st.columns([4, 1], vertical_alignment="center")
with head_left:
    st.title(t("title", lang))
with head_right:
    st.segmented_control(
        t("lang_label", lang),
        options=list(LANGS),
        format_func=LANGS.get,
        key="lang_choice",
        on_change=_on_lang_change,
        label_visibility="collapsed",
    )
st.markdown(t("subtitle", lang))

# ------------------------------ Sidebar form ------------------------------
with st.sidebar:
    st.header(t("sidebar_header", lang))
    st.toggle(t("compact_view", lang), key="compact", help=t("compact_help", lang))

    with st.form("fishing_params"):
        st.subheader(t("location", lang))
        st.text_input(t("spot_name", lang), key="spot_name", help=t("spot_name_help", lang))
        st.text_input(t("url", lang), key="spot_url", help=t("url_help", lang))

        with st.expander(t("tide_loc", lang)):
            st.caption(t("tide_loc_caption", lang))
            c1, c2 = st.columns(2)
            c1.text_input(t("latitude", lang), key="tide_lat", placeholder="38.43")
            c2.text_input(t("longitude", lang), key="tide_lon", placeholder="-9.10")

        for group, group_vars in groupby(VARIABLES, key=lambda v: v.group):
            st.subheader(t(f"group.{group}", lang))
            for var in group_vars:
                short = t(f"var.{var.col}.short", lang)
                help_text = t(f"var.{var.col}.help", lang) or None
                col1, col2 = st.columns(2)
                for column, bound in ((col1, "min"), (col2, "max")):
                    with column:
                        st.number_input(
                            f"{t(bound, lang)} {short} ({var.unit})",
                            min_value=var.input_min,
                            max_value=var.input_max,
                            step=var.step,
                            key=f"{var.col}_{bound}",
                            help=help_text,
                        )

        submitted = st.form_submit_button(t("submit", lang), width='stretch')

if submitted:
    st.session_state.has_results = True
    if st.session_state.compact:
        close_sidebar()

# ------------------------------ Main content ------------------------------
if st.session_state.has_results:
    spot_url: str = st.session_state.spot_url
    user_ranges = {v.col: (st.session_state[f"{v.col}_min"], st.session_state[f"{v.col}_max"])
                   for v in VARIABLES}
    bad = [var_label(v.col) for v in VARIABLES if user_ranges[v.col][0] > user_ranges[v.col][1]]
    if bad:
        st.warning(t("warn_min_max", lang, names=", ".join(bad)))

    try:
        spot_id = parse_spot_id(spot_url)
    except ValueError:
        st.error(t("err_spot_id", lang, url=spot_url))
        st.stop()

    with st.spinner(t("spinner", lang)):
        try:
            df, api_spot_name, api_lat, api_lon = load_forecast(spot_id, spot_url)
        except Exception:  # network trouble shouldn't show a stack trace to users
            logger.exception("Forecast fetch failed for spot %s", spot_id)
            st.error(t("err_fetch", lang))
            st.stop()

    spot_name = st.session_state.spot_name.strip() or api_spot_name or f"Spot {spot_id}"

    if df.empty:
        st.error(t("err_no_data", lang))
        st.stop()

    # ---- Tides (never block the forecast) ----
    tide_curve, tide_extremes = None, None
    coords = resolve_tide_coords(spot_id, st.session_state.tide_lat, st.session_state.tide_lon,
                                 api_lat, api_lon)
    if coords is None:
        st.info(t("info_no_coords", lang))
    else:
        try:
            tide_curve, tide_extremes = load_tides(coords[0], coords[1])
            if tide_curve.empty:
                st.warning(t("warn_tides", lang))
        except Exception:
            logger.exception("Tide fetch failed for %s", coords)
            st.warning(t("warn_tides", lang))

    plot_data = prepare_plot_data(df, user_ranges)

    # ---- Best windows summary ----
    windows = best_windows(plot_data, min_hours=2)
    if windows:
        st.success(t("best_windows", lang) + " · ".join(
            f"{fmt_day_time(t0, lang)} – {fmt_day_time(t1, lang)} ({h} h)" if t1.date() != t0.date()
            else f"{fmt_day_time(t0, lang)}–{t1:%H:%M} ({h} h)" for t0, t1, h in windows[:8]))
    else:
        st.warning(t("no_windows", lang))

    # ---- Week at a glance ----
    st.subheader(t("week", lang))
    rows = _week_rows(plot_data, tide_extremes)
    if st.session_state.compact:
        st.markdown(week_html(rows), unsafe_allow_html=True)
    else:
        st.dataframe(week_table(rows), width='stretch', hide_index=True)

    # ---- Chart ----
    fig = build_figure(plot_data, user_ranges, tide_curve, tide_extremes,
                       title=t("chart.title", lang, name=spot_name), lang=lang)
    if st.session_state.compact:
        # Phone: one day at a time, static image so swiping scrolls the page
        days = sorted({ts.date() for ts in plot_data.status.index.tz_convert(LOCAL_TZ)})
        today = pd.Timestamp.now(tz=LOCAL_TZ).date()
        default_day = today if today in days else days[0]
        if st.session_state.get("chart_day") not in days:
            st.session_state.chart_day = default_day
        day = st.pills(t("pick_day", lang), options=days, format_func=lambda d: fmt_day(d, lang),
                       key="chart_day", label_visibility="collapsed") or default_day
        st.toggle(t("show_models", lang), key="show_models")

        day_start = pd.Timestamp(day).tz_localize(LOCAL_TZ)
        day_fig = build_figure(plot_data.window(day_start, day_start + pd.Timedelta(days=1)),
                               user_ranges, tide_curve, tide_extremes,
                               title=f"{spot_name} – {fmt_day(day, lang)}", lang=lang,
                               compact=True, show_models=st.session_state.show_models)
        st.plotly_chart(day_fig, config={"staticPlot": True})
        st.caption(t("caption_colors_compact", lang))
    else:
        st.plotly_chart(fig, config={"displaylogo": False})
        st.caption(t("caption_colors", lang))
    if tide_curve is not None and not tide_curve.empty and coords:
        st.caption(t("caption_tides", lang, lat=coords[0], lon=coords[1],
                     source=t(f"source.{coords[2]}", lang)))

    st.download_button(
        label=t("download", lang),
        data=fig.to_html(include_plotlyjs="cdn", full_html=True),
        file_name=f"windguru_{spot_name.lower().replace(' ', '_')}.html",
        mime="text/html",
        on_click="ignore",
    )

    # ---- Tables ----
    hours_df = good_hours(plot_data)
    if not hours_df.empty:
        with st.expander(t("good_hours", lang, n=len(hours_df))):
            hours_df["local_time"] = [fmt_day_time(ts, lang) for ts in hours_df["local_time"]]
            hours_df["status"] = [t(f"status.{s}", lang) for s in hours_df["status"]]
            columns = {"local_time": t("col.local_time", lang), "status": t("col.status", lang)}
            columns.update({v.col: var_label(v.col, v.unit) for v in VARIABLES})
            st.dataframe(hours_df.rename(columns=columns), width='stretch', hide_index=True)

    if tide_extremes is not None and not tide_extremes.empty:
        with st.expander(t("tide_table", lang)):
            tt = tide_extremes[(tide_extremes["time"] >= plot_data.start)
                               & (tide_extremes["time"] <= plot_data.end)]
            local = tt["time"].dt.tz_convert(LOCAL_TZ)
            tt = pd.DataFrame({t("col.day", lang): [fmt_day(ts, lang) for ts in local],
                               t("col.time", lang): local.dt.strftime("%H:%M"),
                               t("col.tide", lang): [t(f"tide.{k}", lang) for k in tt["type"]],
                               t("col.height", lang): tt["height_cm"].round(0).astype(int)})
            st.dataframe(tt, width='stretch', hide_index=True)
else:
    # Welcome screen
    st.info(t("welcome_info", lang))
    st.markdown(t("welcome_md", lang))

# Footer
st.divider()
st.markdown(t("footer", lang))

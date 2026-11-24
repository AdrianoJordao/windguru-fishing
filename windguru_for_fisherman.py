"""
    A small app to find good days for fishing based on data provided by Windguru
    Created on:      23-Oct-2025
    Original author: Adriano Jordão
"""

from __future__ import annotations

from itertools import groupby
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from config import (DEFAULT_SPOT_NAME, DEFAULT_SPOT_URL, KNOWN_SPOTS, LOCAL_TZ,
                    VARIABLES)
from forecast_plotly import build_figure
from tides import fetch_tide_curve, find_extremes
from windguru_scraper import (WindguruScraper, best_windows, good_hours,
                              parse_spot_id, prepare_plot_data)

# Page configuration
st.set_page_config(
    page_title="Windguru Fishing Forecast",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 0.5rem !important; }
        h1 { margin-top: 0rem !important; margin-bottom: 0.5rem !important; }
        section[data-testid="stSidebar"] { width: 420px !important; }
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
    return df, scraper.spot_name, scraper.lat, scraper.lon, scraper.messages


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
            st.warning("⚠️ Tide latitude/longitude are not valid numbers. Ignoring them.")
    if payload_lat is not None and payload_lon is not None:
        return payload_lat, payload_lon, "Windguru"
    if spot_id in KNOWN_SPOTS:
        _, lat, lon = KNOWN_SPOTS[spot_id]
        return lat, lon, "built-in"
    return None


# ------------------------------ Session state defaults ------------------------------

for var in VARIABLES:
    st.session_state.setdefault(f"{var.col}_min", var.default[0])
    st.session_state.setdefault(f"{var.col}_max", var.default[1])
st.session_state.setdefault("spot_url", DEFAULT_SPOT_URL)
st.session_state.setdefault("spot_name", DEFAULT_SPOT_NAME)
st.session_state.setdefault("tide_lat", "")
st.session_state.setdefault("tide_lon", "")
st.session_state.setdefault("has_results", False)

# Title
st.title("🎣 Windguru Fishing Forecast")
st.markdown("Find the best days for fishing based on weather conditions")

# ------------------------------ Sidebar form ------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    with st.form("fishing_params"):
        st.subheader("📍 Location")
        st.text_input("Spot Name", key="spot_name", help="Name for your fishing spot")
        st.text_input("Windguru URL", key="spot_url",
                      help="Full URL from windguru.cz (e.g., https://www.windguru.cz/65000)")

        with st.expander("🌊 Tide location (optional)"):
            st.caption("Leave empty to use the spot's coordinates.")
            c1, c2 = st.columns(2)
            c1.text_input("Latitude", key="tide_lat", placeholder="38.43")
            c2.text_input("Longitude", key="tide_lon", placeholder="-9.10")

        for group, group_vars in groupby(VARIABLES, key=lambda v: v.group):
            st.subheader(group)
            for var in group_vars:
                col1, col2 = st.columns(2)
                for column, bound in ((col1, "min"), (col2, "max")):
                    with column:
                        st.number_input(
                            f"{bound.capitalize()} {var.short} ({var.unit})",
                            min_value=var.input_min,
                            max_value=var.input_max,
                            step=var.step,
                            key=f"{var.col}_{bound}",
                            help=var.help,
                        )

        submitted = st.form_submit_button("🎣 Check Forecast", width='stretch')

if submitted:
    st.session_state.has_results = True

# ------------------------------ Main content ------------------------------
if st.session_state.has_results:
    spot_url: str = st.session_state.spot_url
    user_ranges = {v.col: (st.session_state[f"{v.col}_min"], st.session_state[f"{v.col}_max"])
                   for v in VARIABLES}
    bad = [v.label for v in VARIABLES if user_ranges[v.col][0] > user_ranges[v.col][1]]
    if bad:
        st.warning(f"⚠️ Min is larger than max for: {', '.join(bad)}")

    try:
        spot_id = parse_spot_id(spot_url)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    with st.spinner("🌊 Fetching forecast…"):
        try:
            df, api_spot_name, api_lat, api_lon, messages = load_forecast(spot_id, spot_url)
        except Exception as e:  # network trouble shouldn't show a stack trace to users
            st.error(f"❌ Could not fetch the forecast: {e}")
            st.stop()

    spot_name = st.session_state.spot_name.strip() or api_spot_name or f"Spot {spot_id}"

    if df.empty:
        st.error("❌ No forecast data retrieved. Please check the URL and try again.")
        with st.expander("📋 Debug Output"):
            st.text("\n".join(messages))
        st.stop()

    # ---- Tides (never block the forecast) ----
    tide_curve, tide_extremes = None, None
    coords = resolve_tide_coords(spot_id, st.session_state.tide_lat, st.session_state.tide_lon,
                                 api_lat, api_lon)
    if coords is None:
        st.info("ℹ️ No coordinates for this spot, so no tides. "
                "Add a latitude/longitude under *Tide location* in the sidebar.")
    else:
        try:
            tide_curve, tide_extremes = load_tides(coords[0], coords[1])
            if tide_curve.empty:
                st.warning("⚠️ Tide data is unavailable right now.")
        except Exception as e:
            st.warning(f"⚠️ Tide data is unavailable right now ({e}).")

    plot_data = prepare_plot_data(df, user_ranges)

    # ---- Best windows summary ----
    windows = best_windows(plot_data, min_hours=2)
    if windows:
        st.success("**Best windows:** " + " · ".join(
            f"{t0:%a %d/%m %H:%M} – {t1:%a %d/%m %H:%M} ({h} h)" if t1.date() != t0.date()
            else f"{t0:%a %d/%m %H:%M}–{t1:%H:%M} ({h} h)" for t0, t1, h in windows[:8]))
    else:
        st.warning("No stretch of 2+ hours matches all your criteria. Check the yellow hours in the chart.")

    # ---- Chart ----
    fig = build_figure(plot_data, user_ranges, tide_curve, tide_extremes,
                       title=f"Windguru Forecast – {spot_name}")
    st.plotly_chart(fig, config={"displaylogo": False})
    st.caption("🟢 all conditions in range · 🟡 one condition slightly out · 🔴 not recommended. "
               "Click a model in the legend to hide it.")
    if tide_curve is not None and not tide_curve.empty and coords:
        st.caption(f"Tides: Open-Meteo marine model at {coords[0]:.3f}, {coords[1]:.3f} ({coords[2]} coordinates). "
                   "Times are approximate (±30 min); not for navigation.")

    st.download_button(
        label="📥 Download Chart (interactive HTML)",
        data=fig.to_html(include_plotlyjs="cdn", full_html=True),
        file_name=f"windguru_{spot_name.lower().replace(' ', '_')}.html",
        mime="text/html",
        on_click="ignore",
    )

    # ---- Tables ----
    hours_df = good_hours(plot_data)
    if not hours_df.empty:
        with st.expander(f"📋 Good Fishing Hours ({len(hours_df)})"):
            labels = {v.col: f"{v.label} ({v.unit})" for v in VARIABLES}
            st.dataframe(hours_df.rename(columns=labels), width='stretch', hide_index=True)

    if tide_extremes is not None and not tide_extremes.empty:
        with st.expander("🌊 Tide Table"):
            tt = tide_extremes[(tide_extremes["time"] >= plot_data.start)
                               & (tide_extremes["time"] <= plot_data.end)]
            local = tt["time"].dt.tz_convert(LOCAL_TZ)
            tt = pd.DataFrame({"Day": local.dt.strftime("%a %d/%m"),
                               "Time": local.dt.strftime("%H:%M"),
                               "Tide": tt["type"],
                               "Height (cm)": tt["height_cm"].round(0).astype(int)})
            st.dataframe(tt, width='stretch', hide_index=True)

    with st.expander("🔍 Debug Output"):
        st.text("\n".join(messages))
else:
    # Welcome screen
    st.info("👈 Configure your fishing conditions in the sidebar and click '🎣 Check Forecast' to get started!")

    st.markdown("""
    ### How to use:
    1. **Enter your Windguru URL** - Find your spot on windguru.cz and copy the URL
    2. **Set your ideal conditions** - Define acceptable ranges for waves, wind, and weather
    3. **Click Check Forecast** - Get an interactive forecast with color-coded good/bad periods and tides

    ### Color coding:
    - 🟢 **Green**: All conditions within your ranges - perfect for fishing!
    - 🟡 **Yellow**: One condition slightly outside range - still manageable
    - 🔴 **Red**: Multiple conditions outside range - not recommended

    ### Weather variables:
    - 🌧️ **Precipitation**: Rain in mm per hour (typically want 0-2 mm/h)
    - 🌡️ **Temperature**: Air temperature in °C
    - ☁️ **Cloud Cover**: Percentage (some clouds can improve fishing!)
    - 🌊 **Tides**: High and low tides for the spot

    ### Popular spots:
    - Sesimbra: `https://www.windguru.cz/65000`
    - Cascais: `https://www.windguru.cz/574`
    - Costa da Caparica: `https://www.windguru.cz/48963`
    """)

# Footer
st.divider()
st.markdown("Made with ❤️ for fishermen by Adriano Jordão | Data from [Windguru](https://www.windguru.cz) "
            "· Tides from [Open-Meteo](https://open-meteo.com)")

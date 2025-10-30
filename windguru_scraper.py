"""
    A small script to calculate a good day for fishing based on same data provided bu windguru
    Created on:      23-Oct-2025
    Original author: Adriano Jordão (adriano.jordao@gmail.com)
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests
from matplotlib.ticker import NullFormatter

# ---------------- Known model names (fallbacks) ----------------
MODEL_NAMES: Dict[int, str] = {
    3: "GFS",
    21: "ICON",
    37: "ARPEGE",
    43: "ECMWF",
    46: "HARMONIE",
    64: "WRF",
    84: "WW3",
    598: "ICON-EU",
    907: "AROME",
    38: "NAM",
}


@dataclass
class ForecastRecord:
    """ Class that holds the forecast for every hour """
    model: str
    datetime: datetime  # stored as UTC-aware
    wave_height_m: Optional[float]
    wave_period_s: Optional[float]
    wind_speed_kmh: Optional[float]
    wind_gust_kmh: Optional[float]
    # NEW VARIABLES
    precipitation_mm: Optional[float]
    temperature_c: Optional[float]
    cloud_cover_pct: Optional[float]  # Using cloud cover as proxy when tide not available


class WindguruScraper:
    """ The scraper for windguru data """

    def __init__(self, spot_url: str, spot_title_override: Optional[str] = None) -> None:
        self.spot_url: str = spot_url
        self.data: pd.DataFrame | None = None
        self.spot_name: str | None = spot_title_override

    # ------------------------------ Fetching ------------------------------

    def _resolve_model_name(self, model_id: int, payload: dict) -> str:
        for key in ("model_name", "model_name_short", "name", "model"):
            name = payload.get(key)
            if isinstance(name, str) and name.strip():
                return name.strip()
        return MODEL_NAMES.get(model_id, f"Model {model_id}")

    def _maybe_set_spot_name(self, payload: dict, spot_id: str) -> None:
        if self.spot_name:
            return
        for key in ("spot_name", "spot", "location", "name", "spot_name_long"):
            val = payload.get(key)
            if isinstance(val, str) and val.strip():
                self.spot_name = val.strip()
                return
        self.spot_name = f"Spot {spot_id}"

    def fetch_data(self, model_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
        """  fetch data from multiple modules """
        spot_id = self.spot_url.rstrip("/").split("/")[-1]
        print(f"Fetching data for spot {spot_id}…")

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": self.spot_url,
            "Accept": "application/json, text/javascript, */*; q=0.01",
        }

        if model_ids is None:
            model_ids = [3, 37, 907, 64, 21, 43, 46, 598, 84, 38]

        print("Attempting to fetch forecasts from multiple models…")
        all_forecasts: List[dict] = []

        for mid in dict.fromkeys(model_ids):
            complete_url = f"https://www.windguru.net/int/iapi.php?q=forecast&id_model={mid}&id_spot={spot_id}"
            try:
                print(f"  Fetching model {mid}…", end=" ")
                r = requests.get(complete_url, headers=headers, timeout=12)
                if r.status_code != 200:
                    print(f"(HTTP {r.status_code})")
                    continue
                payload = r.json()
                if not payload or "fcst" not in payload:
                    print("(no data)")
                    continue

                self._maybe_set_spot_name(payload, spot_id)
                all_forecasts.append({"model_id": mid, "model_name": self._resolve_model_name(mid, payload), "data": payload})
                print("✓")
            except Exception as e:
                print(f"(error: {e})")

        if not all_forecasts:
            print("\nWarning: No forecast data retrieved. Trying alternative endpoint…")
            alt = f"https://www.windguru.cz/int/iapi.php?q=forecast_spot&id_spot={spot_id}"
            try:
                dbg = requests.get(alt, headers=headers, timeout=12).json()
                print(f"Fallback response keys: {list(dbg)[:8] if isinstance(dbg, dict) else 'n/a'}")
                print(json.dumps(dbg, indent=2)[:600] + "…")
            except Exception as e:
                print(f"Fallback failed: {e}")
            if not self.spot_name:
                self.spot_name = f"Spot {spot_id}"

        return self._parse_api_data(all_forecasts)

    def _parse_api_data(self, forecasts: List[dict]) -> pd.DataFrame:
        print("Parsing forecast data…")
        records: List[ForecastRecord] = []

        for forecast in forecasts:
            model_name: str = forecast["model_name"]
            fcst: dict = forecast["data"]["fcst"]

            initstamp: int = int(fcst.get("initstamp", 0))
            hours: List[int] = [int(h) for h in fcst.get("hours", []) if h is not None]
            if not hours:
                print(f"  Warning: no 'hours' for {model_name}")
                continue

            # Existing variables
            HTSGW = fcst.get("HTSGW", [])
            PERPW = fcst.get("PERPW", [])
            WINDSPD = fcst.get("WINDSPD", [])
            GUST = fcst.get("GUST", [])

            # NEW VARIABLES - multiple possible field names
            APCP = fcst.get("APCP", fcst.get("PRECIP", fcst.get("RAIN", fcst.get("PCPT", fcst.get("APCP", fcst.get("APCP1", []))))))  # precipitation
            TMP = fcst.get("TMP", fcst.get("TMPE", []))  # air temperature
            TCDC = fcst.get("TCDC", [])  # cloud cover (0-100%)

            print(
                f"  {model_name}: {len(hours)}h, "
                f"Wave={len(HTSGW)}, Period={len(PERPW)}, Wind={len(WINDSPD)}, Gust={len(GUST)}, "
                f"Precip={len(APCP)}, Temp={len(TMP)}, Cloud={len(TCDC)}"
            )

            for i, hoff in enumerate(hours):
                try:
                    ts = initstamp + hoff * 3600
                    dt = datetime.fromtimestamp(ts, tz=timezone.utc)

                    def val(arr: list, idx: int) -> Optional[float]:
                        if idx < len(arr):
                            v = arr[idx]
                            if v not in (None, "-", "", "null"):
                                return float(v)
                        return None

                    wave_h = val(HTSGW, i)
                    wave_p = val(PERPW, i)
                    wind_s = val(WINDSPD, i)
                    wind_g = val(GUST, i)
                    precip = val(APCP, i)
                    temp = val(TMP, i)
                    cloud = val(TCDC, i)

                    # knots -> km/h
                    if wind_s is not None:
                        wind_s *= 1.852
                    if wind_g is not None:
                        wind_g *= 1.852

                    records.append(
                        ForecastRecord(
                            model=model_name,
                            datetime=dt,
                            wave_height_m=wave_h,
                            wave_period_s=wave_p,
                            wind_speed_kmh=wind_s,
                            wind_gust_kmh=wind_g,
                            precipitation_mm=precip,
                            temperature_c=temp,
                            cloud_cover_pct=cloud,
                        )
                    )
                except Exception as e:
                    print(f"    Parse error @ hour {i}: {e}")

        self.data = pd.DataFrame([r.__dict__ for r in records])
        print(f"\nExtracted {len(self.data)} forecast rows from {len(forecasts)} models")
        if not self.data.empty:
            self.data["datetime"] = pd.to_datetime(self.data["datetime"], errors="coerce", utc=True)
            print(f"Date range (UTC): {self.data['datetime'].min()}  →  {self.data['datetime'].max()}")
        return self.data

    # ------------------------------ Filtering ------------------------------
    def filter_conditions(
        self,
        wave_height_range: Tuple[float, float] = (0.0, 1.6),
        wave_period_range: Tuple[float, float] = (0.0, 12.0),
        wind_speed_range: Tuple[float, float] = (0.0, 20.0),
        wind_gust_range: Tuple[float, float] = (0.0, 25.0),
        precipitation_range: Tuple[float, float] = (0.0, 5.0),
        temperature_range: Tuple[float, float] = (10.0, 35.0),
        cloud_cover_range: Tuple[float, float] = (0.0, 100.0),
    ) -> pd.DataFrame:
        """ Filter user conditions """

        if self.data is None or self.data.empty:
            print("No data to filter. Fetch data first.")
            return pd.DataFrame()

        data_frame: pd.DataFrame = self.data
        mask = (data_frame["wave_height_m"].between(*wave_height_range)
                & data_frame["wave_period_s"].between(*wave_period_range)
                & data_frame["wind_speed_kmh"].between(*wind_speed_range)
                & data_frame["wind_gust_kmh"].between(*wind_gust_range)
                & data_frame["precipitation_mm"].between(*precipitation_range)
                & data_frame["temperature_c"].between(*temperature_range)
                & data_frame["cloud_cover_pct"].between(*cloud_cover_range))
        filtered: pd.DataFrame = data_frame.loc[mask].copy()
        print(f"\nFiltered to {len(filtered)} records matching criteria.")
        return filtered

    # ------------------------------ Plotting ------------------------------

    def plot_forecast(
        self,
        show_week_from: Optional[datetime] = None,
        figsize: Tuple[int, int] = (20, 16),
        marker_size: float = 3.0,
        line_width: float = 1.0,
        ranges: Dict[str, Tuple[float, float]] | None = None,
        # Styling
        horiz_band_color: str = "#bfdbfe",
        horiz_band_alpha: float = 0.50,
        green_color: str = "#22c55e",
        yellow_color: str = "#ffcd3c",
        red_color: str = "#fca5a5",
        band_alpha: float = 0.65,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Plot forecast with 8 subplots: MEAN + 7 variables (including new ones)
        """
        if self.data is None or self.data.empty:
            print("No data to plot.")
            return
        if not ranges:
            print("No ranges provided; using defaults inside the runner is recommended.")

        tz_pt = ZoneInfo("Europe/Lisbon")

        # --- Prepare base dataframe (UTC) ---
        base = self.data.copy()
        base["datetime"] = pd.to_datetime(base["datetime"], errors="coerce", utc=True)
        base = base.dropna(subset=["datetime"]).sort_values(["model", "datetime"])
        if base.empty:
            print("No valid datetimes to plot after coercion.")
            return

        # --- Determine plotting window ---
        now_pt = datetime.now(tz_pt)
        if show_week_from is None:
            start_local = now_pt.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start_local = show_week_from if show_week_from.tzinfo else show_week_from.replace(tzinfo=tz_pt)
            start_local = start_local.replace(minute=0, second=0, microsecond=0)
        end_local = start_local + timedelta(days=7)

        start_utc = pd.Timestamp(start_local).tz_convert("UTC")
        end_utc = pd.Timestamp(end_local).tz_convert("UTC")
        hourly_index_utc = pd.date_range(start=start_utc, end=end_utc, freq="h", inclusive="left")

        # Clamp window if completely outside data range
        data_min_local = base["datetime"].dt.tz_convert(tz_pt).min()
        data_max_local = base["datetime"].dt.tz_convert(tz_pt).max()
        if end_local <= start_local or end_local < data_min_local or start_local > data_max_local:
            start_local = pd.Timestamp(data_min_local).floor("h").to_pydatetime()
            end_local = (pd.Timestamp(data_max_local).ceil("h") + pd.Timedelta(hours=1)).to_pydatetime()
            start_utc = pd.Timestamp(start_local).tz_convert("UTC")
            end_utc = pd.Timestamp(end_local).tz_convert("UTC")
            hourly_index_utc = pd.date_range(start=start_utc, end=end_utc, freq="H", inclusive="left")
            print("Adjusted x-window to local data range:", start_local, "→", end_local)

        # ---------------------------------------------------------------------------------
        # 1) INTERPOLATE EACH MODEL to hourly
        # ---------------------------------------------------------------------------------
        vars_cols = ["wave_height_m", "wave_period_s", "wind_speed_kmh", "wind_gust_kmh",
                     "precipitation_mm", "temperature_c", "cloud_cover_pct"]

        interpolated: List[pd.DataFrame] = []
        for model, g in base.groupby("model", sort=False):
            g = g.set_index("datetime").sort_index()
            gi = pd.DataFrame(index=hourly_index_utc, columns=vars_cols, dtype="float64")

            for col in vars_cols:
                s = g[col].dropna()
                if s.empty:
                    continue

                t0, t1 = s.index.min(), s.index.max()
                hourly_span = hourly_index_utc[(hourly_index_utc >= t0) & (hourly_index_utc <= t1)]
                idx = s.index.union(hourly_span)

                s_interp = (
                    s.reindex(idx)
                    .interpolate(method="time")
                    .reindex(hourly_span)
                )

                gi.loc[hourly_span, col] = s_interp.values

            gi["model"] = model
            interpolated.append(gi)

        interp_df = pd.concat(interpolated, axis=0)
        interp_df.index.name = "datetime"
        interp_df.reset_index(inplace=True)
        interp_df["dt_local"] = interp_df["datetime"].dt.tz_convert(tz_pt)

        # ---------------------------------------------------------------------------------
        # 2) HOURLY MEAN across models
        # ---------------------------------------------------------------------------------
        means_utc = (
            interp_df
            .set_index("datetime")
            .groupby(level=0)[vars_cols]
            .mean()
            .reindex(hourly_index_utc)
        )
        means_local = means_utc.copy()
        means_local["dt_local"] = means_utc.index.tz_convert(tz_pt)

        # ---------------------------------------------------------------------------------
        # 3) Build vertical bands from MEAN values
        # ---------------------------------------------------------------------------------
        def rel_dev_to_bound(val: float, lo: float, hi: float) -> float:
            if pd.isna(val):
                return float("inf")
            rng = float(hi) - float(lo)
            if rng <= 0:
                return 0.0 if (lo <= val <= hi) else float("inf")
            if lo <= val <= hi:
                return 0.0
            if val > hi:
                return (float(val) - float(hi)) / rng
            else:
                return (float(lo) - float(val)) / rng

        hour_status: Dict[pd.Timestamp, str] = {}
        for ts, row in means_utc.iterrows():
            present_vars = {c: pd.notna(row[c]) for c in vars_cols}
            present_list = [c for c, p in present_vars.items() if p]
            if not present_list:
                hour_status[ts] = "red"
                continue

            ok_map = {c: (ranges[c][0] <= row[c] <= ranges[c][1]) if pd.notna(row[c]) else False for c in present_list}
            n_present = len(present_list)
            n_ok = sum(ok_map.values())

            if n_ok == n_present:
                status = "green"
            elif n_ok == n_present - 1 and n_present >= 1:
                off = [c for c in present_list if not ok_map[c]]
                c = off[0]
                status = "yellow" if rel_dev_to_bound(row[c], *ranges[c]) <= 0.20 else "red"
            else:
                status = "red"

            hour_status[ts] = status

        # Merge consecutive hours
        hourly_spans_local: List[Tuple[datetime, datetime, str]] = []
        cur_status: Optional[str] = None
        span_start: Optional[pd.Timestamp] = None

        def flush_span(sutc: pd.Timestamp, eutc: pd.Timestamp, status: str) -> None:
            t0 = sutc.tz_convert(tz_pt).to_pydatetime()
            t1 = eutc.tz_convert(tz_pt).to_pydatetime()
            hourly_spans_local.append((t0, t1, status))

        for ts in hourly_index_utc:
            s = hour_status.get(ts, "red")
            if cur_status is None:
                cur_status = s
                span_start = ts
            elif s != cur_status:
                flush_span(span_start, ts, cur_status)
                cur_status = s
                span_start = ts
        if cur_status is not None and span_start is not None:
            flush_span(span_start, hourly_index_utc[-1] + pd.Timedelta(hours=1), cur_status)

        # ---------------------------------------------------------------------------------
        # 4) PLOTTING - 8 subplots total
        # ---------------------------------------------------------------------------------
        title_name = self.spot_name or f"Spot {self.spot_url.rstrip('/').split('/')[-1]}"

        fig, axes = plt.subplots(8, 1, figsize=figsize, sharex=True)
        ax_mean, ax_wave_height, ax_wave_period, ax_wind_speed, ax_wind_gust, ax_precipitation, ax_temperature, ax_cloud_cover = axes

        # Vertical bands on all axes
        if hourly_spans_local:
            def paint_vertical(ax) -> None:
                for (t0, t1, status) in hourly_spans_local:
                    color = green_color if status == "green" else (yellow_color if status == "yellow" else red_color)
                    ax.axvspan(t0, t1, color=color, alpha=band_alpha, zorder=0)
            for ax in axes:
                paint_vertical(ax)

        # ---- Mean plot ----
        ax_mean.plot(means_local["dt_local"], means_utc["wave_height_m"], label="Wave height (m)")
        ax_mean.plot(means_local["dt_local"], means_utc["wave_period_s"], label="Wave period (s)")
        ax_mean.plot(means_local["dt_local"], means_utc["wind_speed_kmh"], label="Wind speed (km/h)")
        ax_mean.plot(means_local["dt_local"], means_utc["wind_gust_kmh"], label="Wind gust (km/h)")
        ax_mean.plot(means_local["dt_local"], means_utc["precipitation_mm"], label="Precipitation (mm)")
        ax_mean.plot(means_local["dt_local"], means_utc["temperature_c"], label="Temperature (°C)")
        ax_mean.plot(means_local["dt_local"], means_utc["cloud_cover_pct"], label="Cloud cover (%)")

        # ax_mean.set_ylabel("")
        ax_mean.set_title("Mean of available models", loc="center", pad=8)
        ax_mean.legend(loc="upper right", ncol=3, fontsize=7, frameon=False)
        ax_mean.grid(True, alpha=0.35, zorder=2)

        # ---- Per-variable panes ----
        def plot_variable(ax, col: str, ylabel: str) -> None:
            if ranges and col in ranges:
                ymin, ymax = ranges[col]
                ax.axhspan(ymin, ymax, color=horiz_band_color, alpha=horiz_band_alpha, lw=0, zorder=1)

            for model, g in interp_df.groupby("model", sort=False):
                s = g[col]
                if not s.notna().any():
                    continue

                ax.plot(
                    g["dt_local"],
                    s,
                    label=model,
                    marker="o",
                    markersize=marker_size,
                    linewidth=line_width,
                    alpha=0.9,
                    zorder=5,
                )

            # AFTER
            # ax.set_ylabel("")
            ax.set_title(ylabel, loc="center", pad=6)
            ax.grid(True, alpha=0.35, zorder=2)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="upper left", ncol=2, fontsize=8, frameon=False)

        plot_variable(ax_wave_height, "wave_height_m", "Wave Height (m)")
        plot_variable(ax_wave_period, "wave_period_s", "Wave Period (s)")
        plot_variable(ax_wind_speed, "wind_speed_kmh", "Wind Speed (km/h)")
        plot_variable(ax_wind_gust, "wind_gust_kmh", "Wind Gusts (km/h)")
        plot_variable(ax_precipitation, "precipitation_mm", "Precipitation (mm)")
        plot_variable(ax_temperature, "temperature_c", "Temperature (°C)")
        plot_variable(ax_cloud_cover, "cloud_cover_pct", "Cloud Cover (%)")

        # Wave height subplot mean line
        ax_wave_height.plot(means_local["dt_local"], means_utc["wave_height_m"], color="black", label="Mean (available models)")
        ax_wave_height.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)

        # Wave period subplot mean line
        ax_wave_period.plot(means_local["dt_local"], means_utc["wave_period_s"], color="black", label="Mean (available models)")
        ax_wave_period.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)

        # Wind speed subplot mean line
        ax_wind_speed.plot(means_local["dt_local"], means_utc["wind_speed_kmh"], color="black", label="Mean (available models)")
        ax_wind_speed.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)

        # Wind gust subplot mean line
        ax_wind_gust.plot(means_local["dt_local"], means_utc["wind_gust_kmh"], color="black", label="Mean (available models)")
        ax_wind_gust.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)

        # Precipitation subplot mean line
        ax_precipitation.plot(means_local["dt_local"], means_utc["precipitation_mm"], color="black", label="Mean (available models)")
        ax_precipitation.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)

        # Temperature subplot mean line
        ax_temperature.plot(means_local["dt_local"], means_utc["temperature_c"], color="black", label="Mean (available models)")
        ax_temperature.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)

        # Cloud cover subplot mean line
        ax_cloud_cover.plot(means_local["dt_local"], means_utc["cloud_cover_pct"], color="black", label="Mean (available models)")
        ax_cloud_cover.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)

        # X-lims on all — align to day boundaries so 00:00 is inside the range
        xmin = pd.to_datetime(means_local["dt_local"].min()).floor("D")
        xmax = pd.to_datetime(means_local["dt_local"].max()).ceil("D")
        for ax in axes:
            ax.set_xlim(xmin, xmax)

        # Bottom axis (labels every 3h, explicitly including 00h)
        ax_cloud_cover.xaxis.set_major_locator(
            mdates.HourLocator(byhour=[0, 3, 6, 9, 12, 15, 18, 21], tz=tz_pt)
        )
        ax_cloud_cover.xaxis.set_major_formatter(mdates.DateFormatter("%Hh", tz=tz_pt))

        # Optional: faint minor grid each hour
        ax_cloud_cover.xaxis.set_minor_locator(mdates.HourLocator(interval=1, tz=tz_pt))
        ax_cloud_cover.grid(True, which="minor", alpha=0.12)

        ax_cloud_cover.set_xlabel("Time (hours)")

        # Top axis
        try:
            ax_top = ax_mean.secondary_xaxis("top")
        except AttributeError:
            ax_top = ax_mean.twiny()
        ax_top.set_xlim(ax_mean.get_xlim())
        ax_top.xaxis.set_major_locator(mdates.DayLocator())
        ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%a %m/%d", tz=tz_pt))
        ax_top.tick_params(axis="x", pad=6, labelsize=10)

        fig.suptitle(f"Windguru Forecast – {title_name}", fontsize=16, y=0.995)

        plt.tight_layout(rect=(0, 0, 1, 0.98))
        out_path = save_path or "windguru_forecast.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved as '{out_path}'")

        if show:
            plt.show()
        else:
            plt.close(fig)

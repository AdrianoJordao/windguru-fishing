"""
    A small script to calculate a good day for fishing based on same data provided bu windguru
    Created on:      23-Oct-2025
    Original author: Adriano Jordão (adriano.jordao@gmail.com)
"""

from __future__ import annotations
import json
import re
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
    """ Class that holds the forecast dor every hour """
    model: str
    datetime: datetime  # stored as UTC-aware
    wave_height_m: Optional[float]
    wave_period_s: Optional[float]
    wind_speed_kmh: Optional[float]
    wind_gust_kmh: Optional[float]


def _slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "spot"


class WindguruScraper:
    """ The scraper for windguru data """

    def __init__(self, spot_url: str, spot_title_override: Optional[str] = None) -> None:
        self.spot_url: str = spot_url
        self.data: pd.DataFrame | None = None
        self.spot_name: str | None = spot_title_override  # can be overridden by API if not provided

    # ------------------------------ Fetching ------------------------------

    def _resolve_model_name(self, model_id: int, payload: dict) -> str:
        for key in ("model_name", "model_name_short", "name", "model"):
            name = payload.get(key)
            if isinstance(name, str) and name.strip():
                return name.strip()
        return MODEL_NAMES.get(model_id, f"Model {model_id}")

    def _maybe_set_spot_name(self, payload: dict, spot_id: str) -> None:
        if self.spot_name:  # already forced by caller or set earlier
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

        for mid in dict.fromkeys(model_ids):  # de-dup keep order
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

            HTSGW = fcst.get("HTSGW", [])
            PERPW = fcst.get("PERPW", [])
            WINDSPD = fcst.get("WINDSPD", [])
            GUST = fcst.get("GUST", [])

            print(
                f"  {model_name}: {len(hours)}h, "
                f"Wave Height={len(HTSGW)}, Wave period={len(PERPW)}, Wind Speed={len(WINDSPD)}, Wind Gust={len(GUST)}"
            )

            for i, hoff in enumerate(hours):
                try:
                    ts = initstamp + hoff * 3600
                    # Store as UTC-aware datetime to avoid DST ambiguity later
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
    ) -> pd.DataFrame:
        """ Filter user conditions """

        if self.data is None or self.data.empty:
            print("No data to filter. Fetch data first.")
            return pd.DataFrame()

        data_frame: pd.DataFrame = self.data
        mask = (data_frame["wave_height_m"].between(*wave_height_range)
                & data_frame["wave_period_s"].between(*wave_period_range)
                & data_frame["wind_speed_kmh"].between(*wind_speed_range)
                & data_frame["wind_gust_kmh"].between(*wind_gust_range))
        filtered = data_frame.loc[mask].copy()
        print(f"\nFiltered to {len(filtered)} records matching criteria.")
        return filtered

    # ------------------------------ Plotting ------------------------------

    def plot_forecast(
        self,
        show_week_from: Optional[datetime] = None,
        figsize: Tuple[int, int] = (20, 13),
        marker_size: float = 3.0,
        line_width: float = 1.0,
        ranges: Dict[str, Tuple[float, float]] | None = None,
        # Styling
        horiz_band_color: str = "#bfdbfe",   # pale blue
        horiz_band_alpha: float = 0.50,
        green_color: str = "#22c55e",        # really green
        yellow_color: str = "#ffcd3c",       # lively yellow
        red_color: str = "#fca5a5",          # light red
        band_alpha: float = 0.65,
        save_path: Optional[str] = None,
        show: bool = True,                   # BLOCKING: show one-by-one
    ) -> None:
        """
        1) For each model, resample to HOURLY from start->end of the displayed week (UTC):
        - reindex to hourly DatetimeIndex
        - interpolate(method='time') then bfill/ffill to edges
        2) Per-hour MEAN across models (for each variable). New first subplot shows the four
        mean lines (one per variable).
        3) Vertical color bands are computed from those MEAN values:
        - GREEN  : all present variables' means within thresholds
        - YELLOW : exactly one present variable's mean outside but within 20% of nearest bound
        - RED    : otherwise (incl. zero variables present)
        4) Variable subplots (4) still show each model's line, now using interpolated series.
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

        # --- Determine plotting window (local) and corresponding UTC hourly index ---
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

        # Clamp window if completely outside data range (reuse original behavior)
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
        # 1) INTERPOLATE EACH MODEL to hourly over the window (UTC) — NO EXTRAPOLATION
        #    • Interpolate per variable
        #    • Only between that variable's first/last known timestamps for the model
        #    • Outside that span -> NaN (not present)
        # ---------------------------------------------------------------------------------
        vars_cols = ["wave_height_m", "wave_period_s", "wind_speed_kmh", "wind_gust_kmh"]

        interpolated: List[pd.DataFrame] = []
        for model, g in base.groupby("model", sort=False):
            g = g.set_index("datetime").sort_index()

            # Result container on the global hourly index (all NaN by default)
            gi = pd.DataFrame(index=hourly_index_utc, columns=vars_cols, dtype="float64")

            for col in vars_cols:
                s = g[col].dropna()
                if s.empty:
                    # model has no data for this variable anywhere -> keep NaN
                    continue

                # Limit to this variable's data span
                t0, t1 = s.index.min(), s.index.max()

                # Build an index that includes both original points and hourly points within [t0, t1]
                hourly_span = hourly_index_utc[(hourly_index_utc >= t0) & (hourly_index_utc <= t1)]
                idx = s.index.union(hourly_span)

                # Interpolate in time on the combined index, then select only hourly points in-span
                s_interp = (
                    s.reindex(idx)
                    .interpolate(method="time")   # fills only between known points
                    .reindex(hourly_span)         # keep hourly points within [t0, t1]
                )

                # Write into the model's hourly frame (outside [t0, t1] stays NaN)
                gi.loc[hourly_span, col] = s_interp.values

            gi["model"] = model
            interpolated.append(gi)

        interp_df = pd.concat(interpolated, axis=0)
        interp_df.index.name = "datetime"
        interp_df.reset_index(inplace=True)   # 'datetime' remains UTC-aware
        interp_df["dt_local"] = interp_df["datetime"].dt.tz_convert(tz_pt)

        # ---------------------------------------------------------------------------------
        # 2) HOURLY MEAN across models (UTC)
        # ---------------------------------------------------------------------------------
        means_utc = (
            interp_df
            .set_index("datetime")
            .groupby(level=0)[vars_cols]
            .mean()
            .reindex(hourly_index_utc)   # ensure full hourly coverage
        )
        means_local = means_utc.copy()
        means_local["dt_local"] = means_utc.index.tz_convert(tz_pt)

        # ---------------------------------------------------------------------------------
        # 3) Build vertical bands from MEAN values
        # ---------------------------------------------------------------------------------
        def rel_dev_to_bound(val: float, lo: float, hi: float) -> float:
            """
            Relative deviation to the nearest bound, normalized ONLY by the variable's range (hi - lo).
            Returns 0 if inside range. If hi==lo, treat any outside value as +inf and inside as 0.
            """
            if pd.isna(val):
                return float("inf")
            rng = float(hi) - float(lo)
            if rng <= 0:
                # degenerate range; inside -> 0, outside -> inf
                return 0.0 if (lo <= val <= hi) else float("inf")
            if lo <= val <= hi:
                return 0.0
            # distance to the nearest bound divided by range
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
                c = off[0]  # single offender
                status = "yellow" if rel_dev_to_bound(row[c], *ranges[c]) <= 0.20 else "red"
            else:
                status = "red"

            hour_status[ts] = status

        # Merge consecutive hours with same status; convert span edges to local
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
                flush_span(span_start, ts, cur_status)  # type: ignore[arg-type]
                cur_status = s
                span_start = ts
        if cur_status is not None and span_start is not None:
            flush_span(span_start, hourly_index_utc[-1] + pd.Timedelta(hours=1), cur_status)

        # ---------------------------------------------------------------------------------
        # 4) PLOTTING
        # ---------------------------------------------------------------------------------
        title_name = self.spot_name or f"Spot {self.spot_url.rstrip('/').split('/')[-1]}"

        # 5 subplots: [MEAN] + 4 variable panes
        fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)
        ax_mean, ax_h, ax_p, ax_ws, ax_wg = axes

        # Vertical bands first (background) on all axes
        if hourly_spans_local:
            def paint_vertical(ax) -> None:
                for (t0, t1, status) in hourly_spans_local:
                    color = green_color if status == "green" else (yellow_color if status == "yellow" else red_color)
                    ax.axvspan(t0, t1, color=color, alpha=band_alpha, zorder=0)
            for ax in axes:
                paint_vertical(ax)

        # ---- Mean plot (one line per variable) ----
        # (Units differ; legend clarifies)
        ax_mean.plot(means_local["dt_local"], means_utc["wave_height_m"], label="Mean wave_height (m)", marker="o", markersize=marker_size, linewidth=line_width, alpha=0.9, zorder=5)
        ax_mean.plot(means_local["dt_local"], means_utc["wave_period_s"], label="Mean wave_period (s)", marker="o", markersize=marker_size, linewidth=line_width, alpha=0.9, zorder=5)
        ax_mean.plot(means_local["dt_local"], means_utc["wind_speed_kmh"], label="Mean wind_speed (km/h)", marker="o", markersize=marker_size, linewidth=line_width, alpha=0.9, zorder=5)
        ax_mean.plot(means_local["dt_local"], means_utc["wind_gust_kmh"], label="Mean wind_gust (km/h)", marker="o", markersize=marker_size, linewidth=line_width, alpha=0.9, zorder=5)
        ax_mean.set_ylabel("Hourly mean of available models")
        ax_mean.legend(loc="upper left", ncol=2, fontsize=9, frameon=False)
        ax_mean.grid(True, alpha=0.35, zorder=2)

        # ---- Per-variable panes (each model’s interpolated line) ----
        def plot_variable(ax, col: str, ylabel: str) -> None:
            # horizontal threshold band
            if ranges and col in ranges:
                ymin, ymax = ranges[col]
                ax.axhspan(ymin, ymax, color=horiz_band_color, alpha=horiz_band_alpha, lw=0, zorder=1)

            for model, g in interp_df.groupby("model", sort=False):
                s = g[col]

                # ---- key change: skip models with no data for this variable ----
                if not s.notna().any():        # all NaN -> nothing to plot, no legend entry
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

            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.35, zorder=2)

            # Optional: make sure legend only contains visible handles
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, loc="upper left", ncol=2, fontsize=9, frameon=False)

        plot_variable(ax_h, "wave_height_m", "Wave Height (m)")
        plot_variable(ax_p, "wave_period_s", "Wave Period (s)")
        plot_variable(ax_ws, "wind_speed_kmh", "Wind Speed (km/h)")
        plot_variable(ax_wg, "wind_gust_kmh", "Wind Gusts (km/h)")

        # X-lims on all
        for ax in axes:
            ax.set_xlim(start_local, end_local)

        # Bottom (hours) on last subplot
        ax_wg.xaxis.set_major_locator(mdates.DayLocator())
        ax_wg.xaxis.set_major_formatter(NullFormatter())  # hide day labels at bottom
        ax_wg.xaxis.set_minor_locator(mdates.HourLocator(interval=3))
        ax_wg.xaxis.set_minor_formatter(mdates.DateFormatter("%Hh", tz=tz_pt))
        ax_wg.tick_params(axis="x", which="minor", pad=2)
        ax_wg.set_xlabel("Time (hours)")

        # Top (days) on first subplot
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
        plt.savefig(out_path, dpi=170, bbox_inches="tight")
        print(f"\nPlot saved as '{out_path}'")

        if show:
            plt.show()  # BLOCKS for this beach
        else:
            plt.close(fig)


# # ----------------------------- Runner (multiple spots) -----------------------------
# if __name__ == "__main__":
#     # Thresholds (used for horizontal bands + hourly status logic)
#     user_ranges = {
#         "wave_height_m": (0.0, 1.6),
#         "wave_period_s": (0.0, 12.0),
#         "wind_speed_kmh": (0.0, 20.0),
#         "wind_gust_kmh": (0.0, 25.0),
#     }

#     # Spots to process (URL, pretty title)
#     spots: List[Tuple[str, str]] = [
#         ("https://www.windguru.cz/65000", "Sesimbra"),
#         # ("https://www.windguru.cz/574", "Cascais"),
#         # ("https://www.windguru.cz/501150", "Alcochete"),
#         # ("https://www.windguru.cz/501155", "Praia da Rainha"),
#         # ("https://www.windguru.cz/48963", "Costa da Caparica"),
#     ]

#     for url, title in spots:
#         print("\n" + "=" * 80)
#         print(f"Processing {title}  ({url})")
#         scraper = WindguruScraper(url, spot_title_override=title)

#         try:
#             df = scraper.fetch_data()
#             if df.empty:
#                 print(f"[{title}] No data fetched; skipping plot.")
#                 continue

#             # Optional: filtering (not required for bands/plot)
#             _ = scraper.filter_conditions(
#                 wave_height_range=user_ranges["wave_height_m"],
#                 wave_period_range=user_ranges["wave_period_s"],
#                 wind_speed_range=user_ranges["wind_speed_kmh"],
#                 wind_gust_range=user_ranges["wind_gust_kmh"],
#             )

#             out_file = f"windguru_forecast_{_slugify(title)}.png"
#             scraper.plot_forecast(
#                 ranges=user_ranges,
#                 show_week_from=None,      # anchor to today 00:00 Lisbon by default
#                 figsize=(20, 11),
#                 marker_size=3.0,
#                 line_width=1.0,
#                 # styling (you can tweak these)
#                 horiz_band_color="#bfdbfe",
#                 horiz_band_alpha=0.35,
#                 green_color="#22c55e",
#                 yellow_color="#ffcd3c",
#                 red_color="#fca5a5",      # lighter red
#                 band_alpha=0.65,
#                 save_path=out_file,
#                 show=True,                # BLOCKING: one-by-one windows
#             )
#         except Exception as e:
#             print(f"[{title}] Error: {e}")

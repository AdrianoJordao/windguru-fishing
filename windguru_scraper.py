"""
    Fetch Windguru forecasts for a spot and work out good fishing hours.
    Created on:      23-Oct-2025
    Original author: Adriano Jordão
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from config import DEFAULT_RANGES, LOCAL_TZ, VARIABLE_COLS

logger = logging.getLogger(__name__)

API_URL = "https://www.windguru.net/int/iapi.php"

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
DEFAULT_MODEL_IDS: Tuple[int, ...] = (3, 37, 907, 64, 21, 43, 46, 598, 84, 38)

# Precipitation field names, best first. APCP1 is 1-hour rain; the others
# are accumulated over the model's time step (often 3 h).
PRECIP_FIELDS_HOURLY = ("APCP1",)
PRECIP_FIELDS_STEP = ("APCP", "PRECIP", "RAIN", "PCPT")

KNOTS_TO_KMH = 1.852
YELLOW_TOLERANCE = 0.20   # one variable may be off by up to 20% of its range


def parse_spot_id(spot: str) -> str:
    """ '65000', 'https://www.windguru.cz/65000', '.../65000#tide' -> '65000' """
    text = str(spot).strip()
    path = re.split(r"[?#]", text, maxsplit=1)[0].rstrip("/")
    match = re.search(r"(\d+)$", path) or re.search(r"(\d+)", path)
    if not match:
        raise ValueError(f"Could not find a Windguru spot id in '{spot}'")
    return match.group(1)


@dataclass
class ForecastRecord:
    """ The forecast for one model at one hour """
    model: str
    datetime: datetime  # stored as UTC-aware
    wave_height_m: Optional[float]
    wave_period_s: Optional[float]
    wind_speed_kmh: Optional[float]
    wind_gust_kmh: Optional[float]
    precipitation_mm: Optional[float]   # mm per hour
    temperature_c: Optional[float]
    cloud_cover_pct: Optional[float]


def _num(arr: Sequence, idx: int) -> Optional[float]:
    if idx < len(arr):
        v = arr[idx]
        if v not in (None, "-", "", "null"):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def _precip_per_hour(fcst: dict, hours: List[int]) -> List[Optional[float]]:
    """ Precipitation as mm/h, whichever field the model provides """
    for key in PRECIP_FIELDS_HOURLY:
        arr = fcst.get(key)
        if arr:
            return [_num(arr, i) for i in range(len(hours))]

    for key in PRECIP_FIELDS_STEP:
        arr = fcst.get(key)
        if not arr:
            continue
        out: List[Optional[float]] = []
        for i in range(len(hours)):
            v = _num(arr, i)
            if v is None:
                out.append(None)
                continue
            # amount since the previous step; first step uses the next spacing
            if i > 0:
                step = hours[i] - hours[i - 1]
            elif len(hours) > 1:
                step = hours[1] - hours[0]
            else:
                step = 1
            out.append(v / step if step > 0 else v)
        return out

    return [None] * len(hours)


class WindguruScraper:
    """ Fetches and parses Windguru forecasts for one spot """

    def __init__(self, spot_url: str, spot_title_override: Optional[str] = None) -> None:
        self.spot_url: str = spot_url
        self.spot_id: str = parse_spot_id(spot_url)
        self.data: pd.DataFrame | None = None
        self.spot_name: str | None = spot_title_override or None
        self.lat: Optional[float] = None
        self.lon: Optional[float] = None
        self.messages: List[str] = []   # per-instance log, shown in the app's debug panel

    def _log(self, msg: str) -> None:
        self.messages.append(msg)
        logger.info(msg)

    # ------------------------------ Fetching ------------------------------

    @staticmethod
    def _resolve_model_name(model_id: int, payload: dict) -> str:
        for key in ("model_name", "model_name_short", "name", "model"):
            name = payload.get(key)
            if isinstance(name, str) and name.strip():
                return name.strip()
        return MODEL_NAMES.get(model_id, f"Model {model_id}")

    def _maybe_set_spot_meta(self, payload: dict) -> None:
        """ Spot name and coordinates, if the payload has them """
        if not self.spot_name:
            for key in ("spot_name", "spot", "location", "name", "spot_name_long"):
                val = payload.get(key)
                if isinstance(val, str) and val.strip():
                    self.spot_name = val.strip()
                    break

        if self.lat is None or self.lon is None:
            sources = [payload] + [v for v in payload.values() if isinstance(v, dict)]
            for src in sources:
                lat = src.get("lat", src.get("latitude"))
                lon = src.get("lon", src.get("lng", src.get("longitude")))
                try:
                    lat_f, lon_f = float(lat), float(lon)
                except (TypeError, ValueError):
                    continue
                if -90 <= lat_f <= 90 and -180 <= lon_f <= 180:
                    self.lat, self.lon = lat_f, lon_f
                    break

    def _fetch_model(self, session: requests.Session, model_id: int) -> Optional[dict]:
        url = f"{API_URL}?q=forecast&id_model={model_id}&id_spot={self.spot_id}"
        try:
            r = session.get(url, timeout=12)
            if r.status_code != 200:
                self._log(f"  model {model_id}: HTTP {r.status_code}")
                return None
            payload = r.json()
        except (requests.RequestException, ValueError) as e:
            self._log(f"  model {model_id}: error {e}")
            return None
        if not isinstance(payload, dict) or "fcst" not in payload:
            self._log(f"  model {model_id}: no data")
            return None
        self._log(f"  model {model_id}: ✓")
        return payload

    def fetch_data(self, model_ids: Optional[Iterable[int]] = None) -> pd.DataFrame:
        """ Fetch all models (in parallel) and parse them into one DataFrame """
        ids = list(dict.fromkeys(model_ids or DEFAULT_MODEL_IDS))
        self._log(f"Fetching {len(ids)} models for spot {self.spot_id}…")

        with requests.Session() as session:
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": self.spot_url,
                "Accept": "application/json, text/javascript, */*; q=0.01",
            })
            with ThreadPoolExecutor(max_workers=4) as pool:
                payloads = list(pool.map(lambda mid: self._fetch_model(session, mid), ids))

        forecasts: List[dict] = []
        for mid, payload in zip(ids, payloads):
            if payload is None:
                continue
            self._maybe_set_spot_meta(payload)
            forecasts.append({"model_id": mid,
                              "model_name": self._resolve_model_name(mid, payload),
                              "data": payload})

        if not self.spot_name:
            self.spot_name = f"Spot {self.spot_id}"
        if not forecasts:
            self._log("Warning: no forecast data retrieved.")

        return self._parse_api_data(forecasts)

    def _parse_api_data(self, forecasts: List[dict]) -> pd.DataFrame:
        records: List[ForecastRecord] = []

        for forecast in forecasts:
            model_name: str = forecast["model_name"]
            fcst: dict = forecast["data"]["fcst"]

            initstamp = int(fcst.get("initstamp", 0))
            hours = [int(h) for h in fcst.get("hours", []) if h is not None]
            if not hours:
                self._log(f"  Warning: no 'hours' for {model_name}")
                continue

            HTSGW = fcst.get("HTSGW", [])
            PERPW = fcst.get("PERPW", [])
            WINDSPD = fcst.get("WINDSPD", [])
            GUST = fcst.get("GUST", [])
            TMP = fcst.get("TMP") or fcst.get("TMPE") or []
            TCDC = fcst.get("TCDC", [])
            precip = _precip_per_hour(fcst, hours)

            self._log(
                f"  {model_name}: {len(hours)}h, Wave={len(HTSGW)}, Period={len(PERPW)}, "
                f"Wind={len(WINDSPD)}, Gust={len(GUST)}, "
                f"Precip={sum(p is not None for p in precip)}, Temp={len(TMP)}, Cloud={len(TCDC)}"
            )

            for i, hoff in enumerate(hours):
                wind_s = _num(WINDSPD, i)
                wind_g = _num(GUST, i)
                records.append(ForecastRecord(
                    model=model_name,
                    datetime=datetime.fromtimestamp(initstamp + hoff * 3600, tz=timezone.utc),
                    wave_height_m=_num(HTSGW, i),
                    wave_period_s=_num(PERPW, i),
                    wind_speed_kmh=wind_s * KNOTS_TO_KMH if wind_s is not None else None,
                    wind_gust_kmh=wind_g * KNOTS_TO_KMH if wind_g is not None else None,
                    precipitation_mm=precip[i],
                    temperature_c=_num(TMP, i),
                    cloud_cover_pct=_num(TCDC, i),
                ))

        columns = list(ForecastRecord.__dataclass_fields__)
        self.data = pd.DataFrame([asdict(r) for r in records], columns=columns)
        self._log(f"Extracted {len(self.data)} forecast rows from {len(forecasts)} models")
        if not self.data.empty:
            self.data["datetime"] = pd.to_datetime(self.data["datetime"], errors="coerce", utc=True)
            for col in VARIABLE_COLS:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
            self._log(f"Date range (UTC): {self.data['datetime'].min()} → {self.data['datetime'].max()}")
        return self.data


# =====================================================================
# Analysis: hourly interpolation, mean across models, good/bad hours
# =====================================================================

@dataclass
class PlotData:
    """ Everything the chart and the 'good hours' table need """
    interp: pd.DataFrame          # per-model hourly values: datetime (UTC), model, <vars>
    means: pd.DataFrame           # hourly mean across models, UTC index
    status: pd.Series             # "green" / "yellow" / "red" per hour, UTC index
    spans: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = field(default_factory=list)  # merged, UTC
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None


def _rel_dev_to_bound(val: float, lo: float, hi: float) -> float:
    rng = float(hi) - float(lo)
    if lo <= val <= hi:
        return 0.0
    if rng <= 0:
        return float("inf")
    return (val - hi) / rng if val > hi else (lo - val) / rng


def classify_hours(means: pd.DataFrame,
                   ranges: Dict[str, Tuple[float, float]]) -> pd.Series:
    """
    green:  every available variable is inside its range
    yellow: exactly one is outside, by at most 20% of that range's width
    red:    anything else, or no data at all
    """
    cols = [c for c in ranges if c in means.columns]
    status = []
    for _, row in means.iterrows():
        present = [c for c in cols if pd.notna(row[c])]
        if not present:
            status.append("red")
            continue
        off = [c for c in present if not (ranges[c][0] <= row[c] <= ranges[c][1])]
        if not off:
            status.append("green")
        elif len(off) == 1 and _rel_dev_to_bound(row[off[0]], *ranges[off[0]]) <= YELLOW_TOLERANCE:
            status.append("yellow")
        else:
            status.append("red")
    return pd.Series(status, index=means.index, dtype="object")


def _merge_spans(status: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    spans: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    one_hour = pd.Timedelta(hours=1)
    for ts, s in status.items():
        if spans and spans[-1][2] == s and spans[-1][1] == ts:
            spans[-1] = (spans[-1][0], ts + one_hour, s)
        else:
            spans.append((ts, ts + one_hour, s))
    return spans


def prepare_plot_data(df: pd.DataFrame,
                      ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                      start: Optional[datetime] = None,
                      days: int = 7,
                      tz: tzinfo = LOCAL_TZ) -> PlotData:
    """ Interpolate each model to hourly, average across models and classify each hour """
    ranges = ranges or DEFAULT_RANGES
    base = df.copy()
    base["datetime"] = pd.to_datetime(base["datetime"], errors="coerce", utc=True)
    base = base.dropna(subset=["datetime"]).sort_values(["model", "datetime"])
    if base.empty:
        empty = pd.DataFrame(columns=VARIABLE_COLS)
        return PlotData(pd.DataFrame(columns=["datetime", "model", *VARIABLE_COLS]),
                        empty, pd.Series(dtype="object"))

    # --- plotting window: local midnight today (or `start`) + `days` ---
    if start is None:
        start_local = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        start_local = (start if start.tzinfo else start.replace(tzinfo=tz)).replace(minute=0, second=0, microsecond=0)
    start_utc = pd.Timestamp(start_local).tz_convert("UTC")
    end_utc = start_utc + pd.Timedelta(days=days)

    data_min, data_max = base["datetime"].min(), base["datetime"].max()
    if end_utc < data_min or start_utc > data_max:
        start_utc = data_min.floor("h")
        end_utc = data_max.ceil("h") + pd.Timedelta(hours=1)
    hourly_index = pd.date_range(start=start_utc, end=end_utc, freq="h", inclusive="left")

    # --- 1) interpolate each model to the hourly grid (no extrapolation) ---
    interpolated: List[pd.DataFrame] = []
    for model, g in base.groupby("model", sort=False):
        g = g.set_index("datetime").sort_index()
        g = g[~g.index.duplicated(keep="first")]
        gi = pd.DataFrame(index=hourly_index, columns=VARIABLE_COLS, dtype="float64")
        for col in VARIABLE_COLS:
            s = g[col].dropna()
            if s.empty:
                continue
            span = hourly_index[(hourly_index >= s.index.min()) & (hourly_index <= s.index.max())]
            if span.empty:
                continue
            gi.loc[span, col] = s.reindex(s.index.union(span)).interpolate(method="time").reindex(span).values
        gi["model"] = model
        interpolated.append(gi)

    interp = pd.concat(interpolated, axis=0)
    interp.index.name = "datetime"
    interp = interp.reset_index()

    # --- 2) hourly mean across models ---
    means = interp.groupby("datetime")[VARIABLE_COLS].mean().reindex(hourly_index)
    means.index.name = "datetime"

    # --- 3) classify and merge consecutive hours ---
    status = classify_hours(means, ranges)
    return PlotData(interp=interp, means=means, status=status, spans=_merge_spans(status),
                    start=hourly_index[0], end=hourly_index[-1] + pd.Timedelta(hours=1))


def good_hours(plot_data: PlotData, tz: tzinfo = LOCAL_TZ,
               include_yellow: bool = True) -> pd.DataFrame:
    """ One row per green (and yellow) hour, using the same mean as the chart """
    keep = ["green", "yellow"] if include_yellow else ["green"]
    mask = plot_data.status.isin(keep)
    out = plot_data.means.loc[mask, VARIABLE_COLS].round(1)
    out.insert(0, "status", plot_data.status[mask])
    out.insert(0, "local_time", out.index.tz_convert(tz).strftime("%a %d/%m %H:%M"))
    return out.reset_index(drop=True)


def best_windows(plot_data: PlotData, min_hours: int = 2,
                 tz: tzinfo = LOCAL_TZ) -> List[Tuple[datetime, datetime, int]]:
    """ Green stretches of at least `min_hours`, as local (start, end, hours) """
    out = []
    for t0, t1, s in plot_data.spans:
        hours = int((t1 - t0) / timedelta(hours=1))
        if s == "green" and hours >= min_hours:
            out.append((t0.tz_convert(tz).to_pydatetime(), t1.tz_convert(tz).to_pydatetime(), hours))
    return out

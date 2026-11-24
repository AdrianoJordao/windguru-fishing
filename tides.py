"""
    Tide predictions from the Open-Meteo Marine API (free, no key).

    Open-Meteo's ``sea_level_height_msl`` is a model value (tides plus surge) on a
    grid of a few km, so near the coast the times are typically within about
    15-30 min and the heights are approximate. It is fine for planning a fishing
    trip, but don't use it for navigation.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"


def fetch_tide_curve(lat: float, lon: float, days: int = 8,
                     session: Optional[requests.Session] = None,
                     timeout: float = 10.0) -> pd.Series:
    """
    Hourly tide height (cm) for a location as a UTC-indexed Series.
    Returns an empty Series if the request fails or the API has no sea data there.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "sea_level_height_msl",
        "timezone": "GMT",
        "past_days": 1,          # so "today 00:00" is always covered
        "forecast_days": days,
        "cell_selection": "sea",
    }
    http = session or requests
    try:
        r = http.get(MARINE_URL, params=params, timeout=timeout)
        r.raise_for_status()
        hourly = r.json().get("hourly", {})
    except (requests.RequestException, ValueError) as e:
        logger.warning("Tide request failed: %s", e)
        return pd.Series(dtype="float64")

    times = hourly.get("time") or []
    values = hourly.get("sea_level_height_msl") or []
    if not times or len(times) != len(values):
        logger.warning("Tide response has no usable data")
        return pd.Series(dtype="float64")

    index = pd.to_datetime(times, utc=True)
    curve = pd.Series(pd.to_numeric(values, errors="coerce"), index=index, dtype="float64") * 100.0
    curve.index.name = "datetime"
    if curve.isna().all():
        logger.warning("Tide data is empty for %.3f, %.3f (land cell?)", lat, lon)
        return pd.Series(dtype="float64")
    return curve


def find_extremes(curve: pd.Series) -> pd.DataFrame:
    """
    High and low tides found in a regularly sampled tide curve.

    Each local max/min is refined with a parabola through it and its two
    neighbours, so the time isn't snapped to the sample grid.
    Returns columns: time (UTC), type ("High"/"Low"), height_cm.
    """
    columns = ["time", "type", "height_cm"]
    s = curve.dropna()
    if len(s) < 3:
        return pd.DataFrame(columns=columns)

    t = s.index
    y = s.to_numpy(dtype=float)
    rows = []
    for i in range(1, len(y) - 1):
        left, mid, right = y[i - 1], y[i], y[i + 1]
        if mid >= left and mid > right:
            kind = "High"
        elif mid <= left and mid < right:
            kind = "Low"
        else:
            continue

        step_before = t[i] - t[i - 1]
        step_after = t[i + 1] - t[i]
        time, height = t[i], mid
        denom = left - 2.0 * mid + right
        if step_before == step_after and denom != 0:
            offset = 0.5 * (left - right) / denom          # in samples, within (-1, 1)
            offset = float(np.clip(offset, -1.0, 1.0))
            time = t[i] + offset * step_after
            height = mid - 0.25 * (left - right) * offset
        rows.append((time, kind, float(height)))

    return pd.DataFrame(rows, columns=columns)

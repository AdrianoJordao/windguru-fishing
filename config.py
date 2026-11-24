"""
    Shared configuration: forecast variables, default fishing ranges and known spots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Europe/Lisbon")


@dataclass(frozen=True)
class Variable:
    """ One forecast variable. Display names live in i18n.py (var.<col>.*) """
    col: str                      # column name in the forecast DataFrame
    unit: str
    group: str                    # sidebar section key: waves / wind / weather
    default: Tuple[float, float]  # default acceptable range
    input_min: float
    input_max: float
    step: float


VARIABLES: Tuple[Variable, ...] = (
    Variable("wave_height_m", "m", "waves", (0.0, 1.6), 0.0, 10.0, 0.1),
    Variable("wave_period_s", "s", "waves", (0.0, 12.0), 0.0, 30.0, 1.0),
    Variable("wind_speed_kmh", "km/h", "wind", (0.0, 20.0), 0.0, 100.0, 1.0),
    Variable("wind_gust_kmh", "km/h", "wind", (0.0, 25.0), 0.0, 100.0, 1.0),
    Variable("precipitation_mm", "mm/h", "weather", (0.0, 2.0), 0.0, 50.0, 0.5),
    Variable("temperature_c", "°C", "weather", (5.0, 35.0), -10.0, 50.0, 1.0),
    Variable("cloud_cover_pct", "%", "weather", (0.0, 100.0), 0.0, 100.0, 5.0),
)

VARIABLE_COLS = [v.col for v in VARIABLES]
DEFAULT_RANGES: Dict[str, Tuple[float, float]] = {v.col: v.default for v in VARIABLES}

# Windguru spot id -> (name, latitude, longitude). Used for tides when the
# Windguru payload doesn't carry coordinates. Points are just offshore.
KNOWN_SPOTS: Dict[str, Tuple[str, float, float]] = {
    "65000": ("Sesimbra", 38.43, -9.10),
    "574": ("Cascais", 38.69, -9.42),
    "48963": ("Costa da Caparica", 38.64, -9.25),
}

DEFAULT_SPOT_URL = "https://www.windguru.cz/65000"
DEFAULT_SPOT_NAME = "Sesimbra"

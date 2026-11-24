"""
    Shared configuration: forecast variables, default fishing ranges and known spots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Europe/Lisbon")


@dataclass(frozen=True)
class Variable:
    """ One forecast variable, as shown in the sidebar and in the chart """
    col: str                      # column name in the forecast DataFrame
    label: str                    # chart title
    unit: str
    group: str                    # sidebar section
    short: str                    # short name for the sidebar inputs ("Min <short> (<unit>)")
    default: Tuple[float, float]  # default acceptable range
    input_min: float
    input_max: float
    step: float
    help: Optional[str] = None


VARIABLES: Tuple[Variable, ...] = (
    Variable("wave_height_m", "Wave Height", "m", "🌊 Wave Conditions", "Height",
             (0.0, 1.6), 0.0, 10.0, 0.1),
    Variable("wave_period_s", "Wave Period", "s", "🌊 Wave Conditions", "Period",
             (0.0, 12.0), 0.0, 30.0, 1.0),
    Variable("wind_speed_kmh", "Wind Speed", "km/h", "💨 Wind Conditions", "Speed",
             (0.0, 20.0), 0.0, 100.0, 1.0),
    Variable("wind_gust_kmh", "Wind Gusts", "km/h", "💨 Wind Conditions", "Gust",
             (0.0, 25.0), 0.0, 100.0, 1.0),
    Variable("precipitation_mm", "Precipitation", "mm/h", "🌧️ Weather Conditions", "Rain",
             (0.0, 2.0), 0.0, 50.0, 0.5, "Rain per hour. Typically want 0-2 mm/h for good fishing"),
    Variable("temperature_c", "Temperature", "°C", "🌧️ Weather Conditions", "Temp",
             (5.0, 35.0), -10.0, 50.0, 1.0),
    Variable("cloud_cover_pct", "Cloud Cover", "%", "🌧️ Weather Conditions", "Clouds",
             (0.0, 100.0), 0.0, 100.0, 5.0, "Some clouds can improve fishing"),
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

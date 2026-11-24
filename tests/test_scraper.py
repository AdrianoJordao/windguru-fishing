from datetime import datetime, timezone

import pandas as pd
import pytest

from config import DEFAULT_RANGES, VARIABLE_COLS
from windguru_scraper import (WindguruScraper, best_windows, classify_hours, good_hours,
                              parse_spot_id, prepare_plot_data)

INIT = int(datetime(2026, 11, 24, 0, tzinfo=timezone.utc).timestamp())


@pytest.mark.parametrize("text", ["65000", "https://www.windguru.cz/65000", "https://www.windguru.cz/65000/",
                                  "https://www.windguru.cz/65000#tide", "https://www.windguru.cz/65000?x=1"])
def test_parse_spot_id(text):
    assert parse_spot_id(text) == "65000"


def test_parse_spot_id_invalid():
    with pytest.raises(ValueError):
        parse_spot_id("https://www.windguru.cz/")


def _payload(**fcst):
    return {"model_name": "GFS", "lat": 38.4, "lon": -9.1, "fcst": {"initstamp": INIT, **fcst}}


def test_parse_units_and_missing_values():
    s = WindguruScraper("https://www.windguru.cz/65000")
    s._maybe_set_spot_meta(_payload())
    assert (s.lat, s.lon) == (38.4, -9.1)

    df = s._parse_api_data([{"model_id": 3, "model_name": "GFS", "data": _payload(
        hours=[0, 3, 6], WINDSPD=[10, "-", None], GUST=[20, 21, 22],
        HTSGW=[1.0, 1.1, 1.2], APCP=[3.0, 6.0, None], TMP=[15, 16, 17])}])
    assert len(df) == 3
    assert df.loc[0, "wind_speed_kmh"] == pytest.approx(18.52)
    assert pd.isna(df.loc[1, "wind_speed_kmh"]) and pd.isna(df.loc[2, "wind_speed_kmh"])
    # 3-hourly accumulation -> mm/h
    assert df.loc[0, "precipitation_mm"] == pytest.approx(1.0)
    assert df.loc[1, "precipitation_mm"] == pytest.approx(2.0)
    assert pd.isna(df.loc[0, "cloud_cover_pct"])
    assert str(df["datetime"].dt.tz) == "UTC"


def test_hourly_precip_field_preferred():
    s = WindguruScraper("65000")
    df = s._parse_api_data([{"model_id": 3, "model_name": "GFS", "data": _payload(
        hours=[0, 1], APCP=[9, 9], APCP1=[0.5, 0.7])}])
    assert df["precipitation_mm"].tolist() == [0.5, 0.7]


def test_classify_hours():
    ranges = {"a": (0.0, 10.0), "b": (0.0, 10.0)}
    means = pd.DataFrame({"a": [5, 11, 15, 11, None], "b": [5, 5, 5, 11, None]})
    assert classify_hours(means, ranges).tolist() == ["green", "yellow", "red", "red", "red"]


def test_classify_ignores_missing_variable():
    means = pd.DataFrame({"a": [5.0], "b": [None]})
    assert classify_hours(means, {"a": (0, 10), "b": (0, 10)}).tolist() == ["green"]


def _frame(models):
    rows = []
    for model, wave, wind in models:
        for h in range(0, 48, 3):
            rows.append({"model": model, "datetime": pd.Timestamp(INIT, unit="s", tz="UTC") + pd.Timedelta(hours=h),
                         **{c: None for c in VARIABLE_COLS},
                         "wave_height_m": wave, "wind_speed_kmh": wind(h) if wind else None})
    return pd.DataFrame(rows)


def test_prepare_plot_data_and_good_hours():
    # A wave-only model and a wind-only model: their mean must still be classified.
    df = _frame([("WW3", 1.0, None), ("GFS", None, lambda h: 10.0 if h < 24 else 40.0)])
    start = datetime(2026, 11, 24, 0, tzinfo=timezone.utc)
    pd_ = prepare_plot_data(df, DEFAULT_RANGES, start=start, days=2)

    assert len(pd_.means) == 48
    assert pd_.status.iloc[0] == "green"
    assert pd_.status.iloc[30] == "red"
    gh = good_hours(pd_)
    assert set(gh["status"]) <= {"green", "yellow"}
    assert len(gh) == int(pd_.status.isin(["green", "yellow"]).sum())

    windows = best_windows(pd_, min_hours=2)
    assert windows and windows[0][2] >= 20

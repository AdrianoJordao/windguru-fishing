import numpy as np
import pandas as pd
import pytest
import requests

import tides
from tides import fetch_tide_curve, find_extremes

PERIOD_H = 12.42


def _sine(hours=72, phase_h=2.3, amp=150.0):
    idx = pd.date_range("2026-11-24", periods=hours, freq="h", tz="UTC")
    t = np.arange(hours, dtype=float)
    return pd.Series(amp * np.cos(2 * np.pi * (t - phase_h) / PERIOD_H), index=idx)


def test_find_extremes_on_sine():
    ext = find_extremes(_sine())
    assert len(ext) == 11  # 72 h of a 12.42 h tide, excluding the edges
    assert list(ext["type"][:2]) == ["High", "Low"]
    assert (ext["type"].iloc[1:].values != ext["type"].iloc[:-1].values).all()  # alternates

    start = pd.Timestamp("2026-11-24", tz="UTC")
    for k, row in enumerate(ext.itertuples()):
        expected = start + pd.Timedelta(hours=2.3 + k * PERIOD_H / 2)
        assert abs((row.time - expected).total_seconds()) < 10 * 60
        assert abs(abs(row.height_cm) - 150) < 5


def test_find_extremes_short_input():
    assert find_extremes(pd.Series(dtype=float)).empty


class _Resp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._payload


def test_fetch_tide_curve_parses(monkeypatch):
    payload = {"hourly": {"time": ["2026-11-24T00:00", "2026-11-24T01:00"],
                          "sea_level_height_msl": [0.5, -0.25]}}
    monkeypatch.setattr(tides.requests, "get", lambda *a, **k: _Resp(payload))
    s = fetch_tide_curve(38.4, -9.1)
    assert s.tolist() == [50.0, -25.0]
    assert str(s.index.tz) == "UTC"


@pytest.mark.parametrize("resp", [
    _Resp({"hourly": {"time": ["2026-11-24T00:00"], "sea_level_height_msl": [None]}}),
    _Resp({}, status=500),
    _Resp({"error": True}),
])
def test_fetch_tide_curve_empty_on_bad_data(monkeypatch, resp):
    monkeypatch.setattr(tides.requests, "get", lambda *a, **k: resp)
    assert fetch_tide_curve(38.4, -9.1).empty


def test_fetch_tide_curve_network_error(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(tides.requests, "get", boom)
    assert fetch_tide_curve(38.4, -9.1).empty

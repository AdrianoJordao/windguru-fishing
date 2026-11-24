from datetime import datetime, timezone

from config import DEFAULT_RANGES
from forecast_plotly import build_figure
from tides import find_extremes
from test_scraper import _frame
from test_tides import _sine
from windguru_scraper import prepare_plot_data


def test_build_figure_smoke():
    df = _frame([("WW3", 1.0, None), ("GFS", None, lambda h: 10.0)])
    pd_ = prepare_plot_data(df, DEFAULT_RANGES, start=datetime(2026, 11, 24, tzinfo=timezone.utc), days=2)
    curve = _sine(hours=48)
    fig = build_figure(pd_, DEFAULT_RANGES, curve, find_extremes(curve), title="Test")
    names = {t.name for t in fig.data}
    assert {"WW3", "GFS", "Mean (all models)", "Tide"} <= names
    assert len(fig.to_html(include_plotlyjs="cdn")) > 1000


def test_build_figure_without_tides():
    df = _frame([("GFS", 1.0, lambda h: 10.0)])
    pd_ = prepare_plot_data(df, DEFAULT_RANGES, start=datetime(2026, 11, 24, tzinfo=timezone.utc), days=2)
    fig = build_figure(pd_, DEFAULT_RANGES, None, None)
    assert any("No tide data" in (a.text or "") for a in fig.layout.annotations)

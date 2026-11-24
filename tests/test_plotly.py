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
    fig = build_figure(pd_, DEFAULT_RANGES, curve, find_extremes(curve), title="Test", lang="en")
    names = {t.name for t in fig.data}
    assert {"WW3", "GFS", "Mean (all models)", "Tide"} <= names
    assert len(fig.to_html(include_plotlyjs="cdn")) > 1000


def test_build_figure_without_tides():
    df = _frame([("GFS", 1.0, lambda h: 10.0)])
    pd_ = prepare_plot_data(df, DEFAULT_RANGES, start=datetime(2026, 11, 24, tzinfo=timezone.utc), days=2)
    fig = build_figure(pd_, DEFAULT_RANGES, None, None, lang="en")
    assert any("No tide data" in (a.text or "") for a in fig.layout.annotations)


def test_build_figure_portuguese_default():
    df = _frame([("GFS", 1.0, lambda h: 10.0)])
    pd_ = prepare_plot_data(df, DEFAULT_RANGES, start=datetime(2026, 11, 24, tzinfo=timezone.utc), days=2)
    curve = _sine(hours=48)
    fig = build_figure(pd_, DEFAULT_RANGES, curve, find_extremes(curve))
    names = {t.name for t in fig.data}
    assert {"Média (todos os modelos)", "Maré", "Preia-mar", "Baixa-mar"} <= names
    texts = [a.text or "" for a in fig.layout.annotations]
    assert any("Altura das ondas" in x for x in texts)
    assert any("Ter 24/11" in x for x in texts)


def test_build_figure_compact_one_day():
    import pandas as pd
    df = _frame([("WW3", 1.0, None), ("GFS", None, lambda h: 10.0)])
    pdata = prepare_plot_data(df, DEFAULT_RANGES, start=datetime(2026, 11, 24, tzinfo=timezone.utc), days=2)
    day = pdata.window(pd.Timestamp("2026-11-24", tz="UTC"), pd.Timestamp("2026-11-25", tz="UTC"))
    curve = _sine(hours=48)

    fig = build_figure(day, DEFAULT_RANGES, curve, find_extremes(curve), compact=True, show_models=False)
    names = {t.name for t in fig.data}
    assert "GFS" not in names and "WW3" not in names
    assert fig.layout.showlegend is False

    fig = build_figure(day, DEFAULT_RANGES, curve, find_extremes(curve), compact=True, show_models=True)
    assert {"GFS", "WW3"} <= {t.name for t in fig.data}

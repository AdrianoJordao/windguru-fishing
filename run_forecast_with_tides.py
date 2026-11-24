#!/usr/bin/env python3
"""
Command-line runner: Windguru forecast + tides -> interactive HTML chart.

Usage:
    python run_forecast_with_tides.py [spot_url_or_id] [-o forecast.html]
"""

import argparse
import logging

from config import DEFAULT_RANGES, DEFAULT_SPOT_URL, KNOWN_SPOTS, LOCAL_TZ
from forecast_plotly import build_figure
from tides import fetch_tide_curve, find_extremes
from windguru_scraper import WindguruScraper, best_windows, prepare_plot_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spot", nargs="?", default=DEFAULT_SPOT_URL, help="Windguru spot URL or id")
    parser.add_argument("-o", "--out", default="forecast.html", help="Output HTML file")
    parser.add_argument("--name", help="Spot name for the chart title")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scraper = WindguruScraper(args.spot, spot_title_override=args.name)
    df = scraper.fetch_data()
    if df.empty:
        print("❌ No forecast data retrieved.")
        return

    # Tides
    curve, extremes = None, None
    if scraper.lat is not None and scraper.lon is not None:
        lat, lon = scraper.lat, scraper.lon
    elif scraper.spot_id in KNOWN_SPOTS:
        _, lat, lon = KNOWN_SPOTS[scraper.spot_id]
    else:
        lat = lon = None
    plot_data = prepare_plot_data(df, DEFAULT_RANGES)
    if lat is not None:
        curve = fetch_tide_curve(lat, lon)
        extremes = find_extremes(curve)
        print(f"\nTides at {lat:.3f}, {lon:.3f}:")
        shown = extremes[(extremes["time"] >= plot_data.start) & (extremes["time"] <= plot_data.end)]
        for _, row in shown.iterrows():
            t = row["time"].tz_convert(LOCAL_TZ)
            print(f"  {t:%a %d/%m %H:%M}  {row['type']:<4}  {row['height_cm']:+.0f} cm")
    else:
        print("No coordinates for this spot; skipping tides.")

    fig = build_figure(plot_data, DEFAULT_RANGES, curve, extremes,
                       title=f"Windguru Forecast – {scraper.spot_name}")
    fig.write_html(args.out, include_plotlyjs="cdn")
    print(f"\n✅ Chart saved to {args.out}")

    print("\nBest windows (2+ green hours):")
    for t0, t1, hours in best_windows(plot_data) or []:
        print(f"  {t0:%a %d/%m %H:%M} – {t1:%a %d/%m %H:%M}  ({hours} h)")


if __name__ == "__main__":
    main()

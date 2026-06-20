from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    AIR_TRAFFIC_FILE,
    EXCHANGE_RATE_FILE,
    FREIGHT_FILE,
    PRICE_FILE,
    RAW_DATA_DIR,
    SHIPPING_FILE,
)


def generate_sample_data(days: int = 240) -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")
    trend = np.linspace(0, 12, days)
    seasonal = 2.5 * np.sin(np.arange(days) / 9)
    freight_shock = np.where((np.arange(days) > 120) & (np.arange(days) < 155), 9, 0)
    currency_shock = np.where((np.arange(days) > 170) & (np.arange(days) < 190), 1.4, 0)

    freight_index = 122 + trend + seasonal + freight_shock + rng.normal(0, 1.8, days)
    fuel_price = 82 + 3 * np.sin(np.arange(days) / 17) + rng.normal(0, 1.2, days)
    usd_inr = 83 + np.linspace(0, 1.8, days) + currency_shock + rng.normal(0, 0.08, days)
    cny_inr = 11.5 + np.linspace(0, 0.25, days) + rng.normal(0, 0.03, days)
    vessel_count = np.round(46 + 5 * np.sin(np.arange(days) / 13) - freight_shock / 3 + rng.normal(0, 2, days)).astype(int)
    avg_delay_hours = 18 + freight_shock * 1.8 + rng.normal(0, 3.5, days)
    avg_transit_days = 10 + avg_delay_hours / 30 + rng.normal(0, 0.4, days)
    flight_count = np.round(28 + 3 * np.sin(np.arange(days) / 11) - freight_shock / 5 + rng.normal(0, 1.5, days)).astype(int)
    avg_flight_delay = 32 + freight_shock * 1.2 + rng.normal(0, 5, days)
    cancelled = np.maximum(0, np.round(1 + freight_shock / 8 + rng.normal(0, 0.8, days))).astype(int)

    price = (
        100
        + trend * 0.65
        + (freight_index - freight_index.mean()) * 0.18
        + (usd_inr - usd_inr.mean()) * 1.9
        + (avg_delay_hours - avg_delay_hours.mean()) * 0.08
        + rng.normal(0, 1.0, days)
    )

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "date": dates,
            "product": "electronics",
            "country_origin": "china",
            "destination_country": "india",
            "price": price.round(2),
        }
    ).to_csv(PRICE_FILE, index=False)
    pd.DataFrame(
        {
            "date": dates,
            "origin_port": "shanghai",
            "destination_port": "nhava_sheva",
            "vessel_count": vessel_count,
            "avg_delay_hours": avg_delay_hours.round(2),
            "avg_transit_days": avg_transit_days.round(2),
        }
    ).to_csv(SHIPPING_FILE, index=False)
    pd.DataFrame(
        {
            "date": dates,
            "origin_airport": "pvg",
            "destination_airport": "del",
            "flight_count": flight_count,
            "avg_delay_minutes": avg_flight_delay.round(2),
            "cancelled_count": cancelled,
        }
    ).to_csv(AIR_TRAFFIC_FILE, index=False)
    pd.DataFrame(
        {
            "date": dates,
            "route": "china-india",
            "freight_index": freight_index.round(2),
            "fuel_price": fuel_price.round(2),
        }
    ).to_csv(FREIGHT_FILE, index=False)
    pd.DataFrame(
        {
            "date": dates,
            "usd_inr": usd_inr.round(4),
            "cny_inr": cny_inr.round(4),
        }
    ).to_csv(EXCHANGE_RATE_FILE, index=False)


if __name__ == "__main__":
    generate_sample_data()
    print(f"Wrote sample CSV files to {Path(RAW_DATA_DIR)}")

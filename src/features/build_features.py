import pandas as pd

from src.config import (
    AIR_TRAFFIC_FILE,
    EXCHANGE_RATE_FILE,
    FEATURE_DATA_FILE,
    FREIGHT_FILE,
    LAGS,
    PRICE_FILE,
    PROCESSED_DATA_DIR,
    ROLLING_WINDOWS,
    SHIPPING_FILE,
    TARGET_COLUMN,
    TRAINING_DATA_FILE,
)
from src.data_ingestion.air_traffic_data import load_air_traffic_data
from src.data_ingestion.exchange_rate_data import load_exchange_rate_data
from src.data_ingestion.freight_data import load_freight_data
from src.data_ingestion.price_data import load_price_data
from src.data_ingestion.shipping_data import load_shipping_data
from src.features.lag_features import add_lag_features
from src.features.rolling_features import add_rolling_features

GROUP_COLUMNS = ["product", "origin_country", "destination_country"]
BASE_NUMERIC_FEATURES = [
    "price",
    "vessel_count",
    "avg_vessel_delay",
    "port_congestion_score",
    "avg_transit_time",
    "cargo_flight_count",
    "avg_flight_delay",
    "cancelled_flights",
    "route_capacity_estimate",
    "container_freight_index",
    "freight_rate_change",
    "fuel_price",
    "usd_inr",
    "cny_inr",
    "exchange_rate_change",
]


def load_and_merge_raw_data(
    price_path=PRICE_FILE,
    shipping_path=SHIPPING_FILE,
    air_path=AIR_TRAFFIC_FILE,
    freight_path=FREIGHT_FILE,
    exchange_path=EXCHANGE_RATE_FILE,
) -> pd.DataFrame:
    price = load_price_data(price_path)
    shipping = load_shipping_data(shipping_path)
    air = load_air_traffic_data(air_path)
    freight = load_freight_data(freight_path)
    exchange = load_exchange_rate_data(exchange_path)

    df = price.merge(shipping, on="date", how="left")
    df = df.merge(air, on="date", how="left")
    df = df.merge(freight, on="date", how="left")
    df = df.merge(exchange, on="date", how="left")
    return df.sort_values(GROUP_COLUMNS + ["date"]).reset_index(drop=True)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(GROUP_COLUMNS + ["date"]).copy()
    grouped = out.groupby(GROUP_COLUMNS, sort=False)
    for horizon in [7, 14, 30]:
        future_price = grouped["price"].shift(-horizon)
        out[f"target_{horizon}d"] = (future_price - out["price"]) / out["price"] * 100
    return out


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price_momentum_7"] = out["price"] - out["price_lag_7"]
    out["freight_momentum_7"] = out["container_freight_index"] - out["container_freight_index_lag_7"]
    out["exchange_rate_momentum_7"] = out["usd_inr"] - out["usd_inr_lag_7"]
    return out


def add_risk_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    congestion_threshold = out["port_congestion_score"].quantile(0.75)
    freight_spike_threshold = out["freight_rate_change"].quantile(0.85)
    capacity_drop_threshold = out["cargo_flight_count"].pct_change().quantile(0.15)
    currency_pressure_threshold = out["exchange_rate_change"].quantile(0.85)

    flight_change = out.groupby(GROUP_COLUMNS, sort=False)["cargo_flight_count"].pct_change().fillna(0)
    out["high_congestion"] = (out["port_congestion_score"] > congestion_threshold).astype(int)
    out["freight_spike"] = (out["freight_rate_change"] > freight_spike_threshold).astype(int)
    out["air_capacity_drop"] = (flight_change < capacity_drop_threshold).astype(int)
    out["currency_pressure"] = (out["exchange_rate_change"] > currency_pressure_threshold).astype(int)
    return out


def build_feature_dataset(save: bool = True) -> pd.DataFrame:
    df = load_and_merge_raw_data()
    df = add_targets(df)
    df = add_lag_features(df, BASE_NUMERIC_FEATURES, GROUP_COLUMNS, LAGS)
    df = add_rolling_features(df, BASE_NUMERIC_FEATURES, GROUP_COLUMNS, ROLLING_WINDOWS)
    df = add_momentum_features(df)
    df = add_risk_signals(df)
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    if save:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        load_and_merge_raw_data().to_csv(TRAINING_DATA_FILE, index=False)
        df.to_csv(FEATURE_DATA_FILE, index=False)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = set(GROUP_COLUMNS + ["date", "target_7d", "target_14d", "target_30d"])
    return [column for column in df.columns if column not in excluded and pd.api.types.is_numeric_dtype(df[column])]


if __name__ == "__main__":
    dataset = build_feature_dataset(save=True)
    print(f"Wrote {len(dataset)} feature rows to {FEATURE_DATA_FILE}")

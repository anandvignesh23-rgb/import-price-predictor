import pandas as pd

from src.config import AIR_TRAFFIC_FILE
from src.utils.time_utils import parse_date_column


def load_air_traffic_data(path=AIR_TRAFFIC_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "flight_count": "cargo_flight_count",
            "avg_delay_minutes": "avg_flight_delay",
            "cancelled_count": "cancelled_flights",
        }
    )
    if "route_capacity_estimate" not in df.columns:
        df["route_capacity_estimate"] = (df["cargo_flight_count"] - df["cancelled_flights"]).clip(lower=0)
    keep = [
        "date",
        "cargo_flight_count",
        "avg_flight_delay",
        "cancelled_flights",
        "route_capacity_estimate",
    ]
    return parse_date_column(df[keep])

import pandas as pd

from src.config import SHIPPING_FILE
from src.utils.time_utils import parse_date_column


def load_shipping_data(path=SHIPPING_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "avg_delay_hours": "avg_vessel_delay",
            "avg_transit_days": "avg_transit_time",
        }
    )
    if "port_congestion_score" not in df.columns:
        delay = df["avg_vessel_delay"].clip(lower=0)
        vessel_count = df["vessel_count"].clip(lower=1)
        df["port_congestion_score"] = (delay / delay.max() * 70 + vessel_count / vessel_count.max() * 30).fillna(0)
    keep = [
        "date",
        "vessel_count",
        "avg_vessel_delay",
        "port_congestion_score",
        "avg_transit_time",
    ]
    return parse_date_column(df[keep])

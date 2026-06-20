import pandas as pd

from src.config import FREIGHT_FILE
from src.utils.time_utils import parse_date_column


def load_freight_data(path=FREIGHT_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"freight_index": "container_freight_index"})
    if "container_freight_index" not in df.columns:
        raise ValueError("Freight data must include freight_index or container_freight_index")
    if "freight_rate_change" not in df.columns:
        df["freight_rate_change"] = df["container_freight_index"].pct_change().fillna(0) * 100
    keep = ["date", "container_freight_index", "freight_rate_change", "fuel_price"]
    return parse_date_column(df[keep])

import pandas as pd

from src.config import EXCHANGE_RATE_FILE
from src.utils.time_utils import parse_date_column


def load_exchange_rate_data(path=EXCHANGE_RATE_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "exchange_rate_change" not in df.columns:
        df["exchange_rate_change"] = df["usd_inr"].pct_change().fillna(0) * 100
    keep = ["date", "usd_inr", "cny_inr", "exchange_rate_change"]
    return parse_date_column(df[keep])

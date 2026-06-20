import pandas as pd

from src.config import PRICE_FILE
from src.utils.time_utils import parse_date_column


def load_price_data(path=PRICE_FILE) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "country_origin": "origin_country",
            "destination": "destination_country",
        }
    )
    required = {"date", "product", "origin_country", "destination_country", "price"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Price data is missing columns: {sorted(missing)}")
    return parse_date_column(df)

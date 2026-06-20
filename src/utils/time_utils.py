import pandas as pd


def parse_date_column(df: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column])
    return out.sort_values(column).reset_index(drop=True)

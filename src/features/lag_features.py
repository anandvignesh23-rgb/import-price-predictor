import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    group_columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    out = df.sort_values(group_columns + ["date"]).copy()
    grouped = out.groupby(group_columns, sort=False)
    lagged_columns = {}
    for column in columns:
        for lag in lags:
            lagged_columns[f"{column}_lag_{lag}"] = grouped[column].shift(lag)
    return pd.concat([out, pd.DataFrame(lagged_columns, index=out.index)], axis=1)

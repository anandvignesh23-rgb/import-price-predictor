import pandas as pd


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    group_columns: list[str],
    windows: list[int],
) -> pd.DataFrame:
    out = df.sort_values(group_columns + ["date"]).copy()
    grouped = out.groupby(group_columns, sort=False)
    rolling_columns = {}
    for column in columns:
        for window in windows:
            shifted = grouped[column].shift(1)
            grouped_shifted = shifted.groupby([out[col] for col in group_columns])
            rolling_columns[f"{column}_roll_mean_{window}"] = grouped_shifted.transform(
                lambda series: series.rolling(window, min_periods=1).mean()
            )
            rolling_columns[f"{column}_roll_std_{window}"] = grouped_shifted.transform(
                lambda series: series.rolling(window, min_periods=2).std()
            )
    return pd.concat([out, pd.DataFrame(rolling_columns, index=out.index)], axis=1)

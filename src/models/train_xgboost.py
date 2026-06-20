import json

import joblib
import pandas as pd
from xgboost import XGBRegressor

from src.config import FEATURE_DATA_FILE, METRICS_FILE, MODEL_FILE, MODEL_DIR, RANDOM_STATE, TARGET_COLUMN
from src.features.build_features import build_feature_dataset, get_feature_columns
from src.models.evaluate import regression_metrics


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True)
    train_end = int(len(df) * 0.7)
    val_end = int(len(df) * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def make_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
    )


def train_model() -> dict:
    df = build_feature_dataset(save=True)
    df = df.dropna().reset_index(drop=True)
    feature_columns = get_feature_columns(df)
    train, val, test = chronological_split(df)

    model = make_model()
    model.fit(
        train[feature_columns],
        train[TARGET_COLUMN],
        eval_set=[(val[feature_columns], val[TARGET_COLUMN])],
        verbose=False,
    )

    metrics = {
        "validation": regression_metrics(val[TARGET_COLUMN], model.predict(val[feature_columns])),
        "test": regression_metrics(test[TARGET_COLUMN], model.predict(test[feature_columns])),
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "target": TARGET_COLUMN,
        "feature_count": len(feature_columns),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
            "latest_feature_row": df.iloc[-1][feature_columns].to_dict(),
        },
        MODEL_FILE,
    )
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    result = train_model()
    print(json.dumps(result, indent=2))

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.config import DEFAULT_DESTINATION_COUNTRY, DEFAULT_ORIGIN_COUNTRY, DEFAULT_PRODUCT, FEATURE_DATA_FILE, MODEL_FILE
from src.features.build_features import GROUP_COLUMNS, build_feature_dataset


def risk_level(predicted_change: float) -> str:
    if predicted_change >= 5:
        return "High"
    if predicted_change >= 2:
        return "Medium"
    return "Low"


@lru_cache(maxsize=1)
def load_model_artifact() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model artifact not found. Run `python -m src.models.train_xgboost` first.")
    return joblib.load(MODEL_FILE)


def latest_feature_row(product: str, origin_country: str, destination_country: str) -> pd.Series:
    if FEATURE_DATA_FILE.exists():
        df = pd.read_csv(FEATURE_DATA_FILE, parse_dates=["date"])
    else:
        df = build_feature_dataset(save=True)
    mask = (
        (df["product"].str.lower() == product.lower())
        & (df["origin_country"].str.lower() == origin_country.lower())
        & (df["destination_country"].str.lower() == destination_country.lower())
    )
    filtered = df.loc[mask].sort_values("date")
    if filtered.empty:
        available = df[GROUP_COLUMNS].drop_duplicates().to_dict(orient="records")
        raise ValueError(f"No feature rows found for route. Available combinations: {available}")
    return filtered.iloc[-1]


def top_feature_factors(model, row: pd.DataFrame, feature_columns: list[str], limit: int = 5) -> list[str]:
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(row[feature_columns])
        scores = np.abs(values[0])
    except Exception:
        scores = getattr(model, "feature_importances_", np.zeros(len(feature_columns)))
    order = np.argsort(scores)[::-1][:limit]
    return [feature_columns[index] for index in order]


def predict_price_change(
    product: str = DEFAULT_PRODUCT,
    origin_country: str = DEFAULT_ORIGIN_COUNTRY,
    destination_country: str = DEFAULT_DESTINATION_COUNTRY,
    horizon: int = 14,
) -> dict:
    if horizon != 14:
        raise ValueError("MVP model currently supports only the 14-day horizon.")

    artifact = load_model_artifact()
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    latest_row = latest_feature_row(product, origin_country, destination_country)
    X = pd.DataFrame([latest_row[feature_columns].to_dict()])
    X = X.apply(pd.to_numeric, errors="raise")
    prediction = float(model.predict(X)[0])

    return {
        "product": product,
        "origin_country": origin_country,
        "destination_country": destination_country,
        "prediction_horizon": "14_days",
        "predicted_price_change_percent": round(prediction, 3),
        "risk_level": risk_level(prediction),
        "main_factors": top_feature_factors(model, X, feature_columns),
    }

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import predict_price_change
from src.models.train_xgboost import train_model

app = FastAPI(
    title="Imported Goods Price Fluctuation Predictor",
    description="Predicts 14-day imported electronics/components price movement for India.",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    product: str = Field(default="electronics")
    origin_country: str = Field(default="china")
    destination_country: str = Field(default="india")
    horizon: int = Field(default=14)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    try:
        return predict_price_change(
            product=request.product,
            origin_country=request.origin_country,
            destination_country=request.destination_country,
            horizon=request.horizon,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/retrain")
def retrain() -> dict:
    try:
        return train_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

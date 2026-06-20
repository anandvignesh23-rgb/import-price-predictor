from fastapi.testclient import TestClient

from scripts.generate_sample_data import generate_sample_data
from src.api.main import app
from src.models.train_xgboost import train_model


def test_health_and_prediction_endpoint():
    generate_sample_data(days=140)
    train_model()
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}
    response = client.post(
        "/predict",
        json={
            "product": "electronics",
            "origin_country": "china",
            "destination_country": "india",
            "horizon": 14,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction_horizon"] == "14_days"
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert len(body["main_factors"]) == 5

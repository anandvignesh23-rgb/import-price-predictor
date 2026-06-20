# Imported Goods Price Fluctuation Predictor

Machine-learning MVP for predicting short-term price movement of imported electronics/components into India. The project uses CSV-based transport, freight, currency, and product price data, builds lag/rolling/momentum features, trains an XGBoost regressor for 14-day price change, and exposes predictions through FastAPI and Streamlit.

## What It Predicts

The current MVP predicts:

- Product: `electronics`
- Origin: `china`
- Destination: `india`
- Horizon: `14_days`

Example API output:

```json
{
  "product": "electronics",
  "origin_country": "china",
  "destination_country": "india",
  "prediction_horizon": "14_days",
  "predicted_price_change_percent": 3.8,
  "risk_level": "Medium",
  "main_factors": [
    "container_freight_index_lag_7",
    "avg_vessel_delay_roll_mean_14",
    "usd_inr_lag_7"
  ]
}
```

## Project Structure

```text
data/
  raw/            CSV inputs
  processed/      merged and feature datasets
notebooks/        exploration placeholders
src/
  data_ingestion/ CSV loaders
  features/       lag, rolling, target, momentum, risk-signal features
  models/         XGBoost training, evaluation, prediction
  api/            FastAPI app
dashboard/        Streamlit dashboard
scripts/          sample data generator
tests/            feature and API smoke tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run The MVP

Generate deterministic sample CSVs:

```bash
python -m scripts.generate_sample_data
```

Train the XGBoost model:

```bash
python -m src.models.train_xgboost
```

Start the API:

```bash
uvicorn src.api.main:app --reload
```

Call prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"product":"electronics","origin_country":"china","destination_country":"india","horizon":14}'
```

Start the dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

## CSV Inputs

Replace the generated files in `data/raw/` with real downloaded data using these schemas:

- `price_data.csv`: `date,product,country_origin,destination_country,price`
- `shipping_data.csv`: `date,origin_port,destination_port,vessel_count,avg_delay_hours,avg_transit_days`
- `air_traffic_data.csv`: `date,origin_airport,destination_airport,flight_count,avg_delay_minutes,cancelled_count`
- `freight_data.csv`: `date,route,freight_index,fuel_price`
- `exchange_rate_data.csv`: `date,usd_inr,cny_inr`

## Model Details

The target is percentage price change after 14 days:

```python
target_14d = (price.shift(-14) - price) / price * 100
```

Features include:

- Lags: `1, 3, 7, 14, 30`
- Rolling means and standard deviations: `3, 7, 14, 30`
- Momentum: price, freight, and USD/INR movement
- Risk signals: high congestion, freight spike, air capacity drop, currency pressure

The train/validation/test split is chronological: 70% / 15% / 15%.

## API

`GET /health`

```json
{"status":"ok"}
```

`POST /predict`

```json
{
  "product": "electronics",
  "origin_country": "china",
  "destination_country": "india",
  "horizon": 14
}
```

`POST /retrain`

Retrains from the latest CSV files and returns validation/test metrics.

## Validation

```bash
pytest
```

Metrics are saved to `models/metrics.json` after training and include MAE, RMSE, MAPE, and directional accuracy.

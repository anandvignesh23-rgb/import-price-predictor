from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"

PRICE_FILE = RAW_DATA_DIR / "price_data.csv"
SHIPPING_FILE = RAW_DATA_DIR / "shipping_data.csv"
AIR_TRAFFIC_FILE = RAW_DATA_DIR / "air_traffic_data.csv"
FREIGHT_FILE = RAW_DATA_DIR / "freight_data.csv"
EXCHANGE_RATE_FILE = RAW_DATA_DIR / "exchange_rate_data.csv"

TRAINING_DATA_FILE = PROCESSED_DATA_DIR / "training_dataset.csv"
FEATURE_DATA_FILE = PROCESSED_DATA_DIR / "feature_dataset.csv"
MODEL_FILE = MODEL_DIR / "xgb_price_model.pkl"
METRICS_FILE = MODEL_DIR / "metrics.json"

DEFAULT_PRODUCT = "electronics"
DEFAULT_ORIGIN_COUNTRY = "china"
DEFAULT_DESTINATION_COUNTRY = "india"
DEFAULT_HORIZON = 14

LAGS = [1, 3, 7, 14, 30]
ROLLING_WINDOWS = [3, 7, 14, 30]
TARGET_COLUMN = "target_14d"
RANDOM_STATE = 42

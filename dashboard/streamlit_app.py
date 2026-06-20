import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DEFAULT_DESTINATION_COUNTRY, DEFAULT_ORIGIN_COUNTRY, DEFAULT_PRODUCT, FEATURE_DATA_FILE, METRICS_FILE
from src.features.build_features import build_feature_dataset
from src.models.predict import predict_price_change

st.set_page_config(page_title="Import Price Predictor", layout="wide")
st.title("Imported Goods Price Fluctuation Predictor")

with st.sidebar:
    product = st.text_input("Product", DEFAULT_PRODUCT)
    origin_country = st.text_input("Origin country", DEFAULT_ORIGIN_COUNTRY)
    destination_country = st.text_input("Destination country", DEFAULT_DESTINATION_COUNTRY)
    horizon = st.selectbox("Prediction horizon", [14], index=0)

if FEATURE_DATA_FILE.exists():
    df = pd.read_csv(FEATURE_DATA_FILE, parse_dates=["date"])
else:
    df = build_feature_dataset(save=True)

prediction = predict_price_change(product, origin_country, destination_country, horizon)

col1, col2, col3 = st.columns(3)
col1.metric("Predicted 14-day change", f"{prediction['predicted_price_change_percent']:.2f}%")
col2.metric("Risk level", prediction["risk_level"])
col3.metric("Top factors", len(prediction["main_factors"]))

route_df = df[
    (df["product"].str.lower() == product.lower())
    & (df["origin_country"].str.lower() == origin_country.lower())
    & (df["destination_country"].str.lower() == destination_country.lower())
].sort_values("date")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.plotly_chart(px.line(route_df, x="date", y="price", title="Price history"), use_container_width=True)
    st.plotly_chart(
        px.line(route_df, x="date", y="port_congestion_score", title="Port congestion trend"),
        use_container_width=True,
    )
with chart_col2:
    st.plotly_chart(
        px.line(route_df, x="date", y="container_freight_index", title="Freight index"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.line(route_df, x="date", y="cargo_flight_count", title="Air cargo trend"),
        use_container_width=True,
    )

st.subheader("Main contributing features")
st.dataframe(pd.DataFrame({"feature": prediction["main_factors"]}), hide_index=True, use_container_width=True)

if METRICS_FILE.exists():
    st.subheader("Latest model metrics")
    st.json(json.loads(METRICS_FILE.read_text()))

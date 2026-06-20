from scripts.generate_sample_data import generate_sample_data
from src.features.build_features import build_feature_dataset, get_feature_columns


def test_feature_dataset_contains_target_and_engineered_features():
    generate_sample_data(days=120)
    df = build_feature_dataset(save=True)
    feature_columns = get_feature_columns(df)

    assert "target_14d" in df.columns
    assert "price_lag_7" in df.columns
    assert "container_freight_index_roll_mean_14" in df.columns
    assert "price_momentum_7" in df.columns
    assert "high_congestion" in df.columns
    assert len(feature_columns) > 20
    assert len(df) > 50

from unittest.mock import patch

import pandas as pd

from prodml.data import split_data
from prodml.features import engineer_features
from prodml.train import main as train_main


def test_engineer_features():
    # Test feature engineering pipeline logic
    df = pd.DataFrame(
        {
            "PULocationID": [10, 20],
            "DOLocationID": [15, 25],
            "trip_distance": [1.2, 3.4],
            "lpep_pickup_datetime": pd.to_datetime(
                ["2023-01-01 10:00:00", "2023-01-01 11:00:00"]
            ),
            "lpep_dropoff_datetime": pd.to_datetime(
                ["2023-01-01 10:15:00", "2023-01-01 11:20:00"]
            ),
        }
    )
    df_engineered = engineer_features(df)
    assert "PU_DO" in df_engineered.columns
    assert "duration" in df_engineered.columns


def test_split_data():
    # Test dataset splitting logic into train and validation sets
    df = pd.DataFrame(
        {
            "PU_DO": ["10_15", "20_25", "30_35", "40_45"],
            "trip_distance": [1.0, 2.0, 3.0, 4.0],
            "duration": [10.0, 12.0, 15.0, 20.0],
        }
    )
    df_train, df_val = split_data(df)
    assert len(df_train) > 0
    assert len(df_val) > 0


@patch("boto3.client")
@patch("prodml.train.load_data")
def test_train_main(mock_load_data, mock_boto_client, tmp_path, monkeypatch):
    # Mock input data frame for training pipeline
    df_mock = pd.DataFrame(
        {
            "PULocationID": [10, 20, 30, 40, 50],
            "DOLocationID": [15, 25, 35, 45, 55],
            "trip_distance": [1.2, 3.4, 2.5, 5.0, 1.1],
            "lpep_pickup_datetime": pd.to_datetime(["2023-01-01 10:00:00"] * 5),
            "lpep_dropoff_datetime": pd.to_datetime(["2023-01-01 10:15:00"] * 5),
        }
    )
    mock_load_data.return_value = df_mock

    # Set up temporary path for model persistence during tests
    fake_model_path = tmp_path / "model.pkl"
    monkeypatch.setattr("prodml.config.settings.MODEL_PATH", fake_model_path)

    # Configure local file store URI for MLflow to avoid external connections
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file://{tmp_path}/mlruns")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")

    # Execute training pipeline main function
    train_main()

    # Assert that the model file was successfully created
    assert fake_model_path.exists()

from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from prodml.inference import load_inference_data, predict_duration


@patch("prodml.inference.load_data")
def test_load_inference_data(mock_load_data):
    df_mock = pd.DataFrame(
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
    mock_load_data.return_value = df_mock

    df_result = load_inference_data("dummy_path", num_samples=2)

    assert len(df_result) == 2
    assert "PU_DO" in df_result.columns


@patch("prodml.inference.mlflow.xgboost.load_model")
@patch("prodml.inference.load_inference_data")
@patch("prodml.inference.mlflow.set_tracking_uri")
def test_predict_duration(mock_set_uri, mock_load_inf_data, mock_load_model):
    # Mock dataframe returned by data loading
    mock_load_inf_data.return_value = pd.DataFrame(
        {"PU_DO": ["10_15", "20_25"], "trip_distance": [1.2, 3.4]}
    )

    # Mock xgboost model and its predict method
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([12.5, 25.0])
    mock_load_model.return_value = mock_model

    # Run the function
    predictions = predict_duration(model_uri="dummy_uri", data_path="dummy_path")

    # Assertions
    assert len(predictions) == 2
    assert predictions[0] == 12.5
    assert predictions[1] == 25.0

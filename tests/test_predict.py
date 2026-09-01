import joblib
from sklearn.linear_model import LinearRegression
from prodml.predict import DurationPredictor


def test_duration_predictor(tmp_path, monkeypatch):
    # Create a temporary dummy model file to satisfy the loading path requirement
    dummy_model_path = tmp_path / "model.pkl"
    model = LinearRegression()
    joblib.dump(model, dummy_model_path)

    # Point the application's model path setting to the temporary file
    monkeypatch.setattr("prodml.config.settings.MODEL_PATH", dummy_model_path)

    # Initialize the predictor using the temporary model path
    predictor = DurationPredictor(model_path=str(dummy_model_path))
    predictor.load()

    # Define sample features for prediction test
    sample_features = {"PU_DO": "130_205", "trip_distance": 3.5}

    # Execute prediction and validate output type and value range
    pred = predictor.predict_one(sample_features)
    assert isinstance(pred, float)
    assert pred >= 0
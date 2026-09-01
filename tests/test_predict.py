from unittest.mock import patch

import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from prodml.predict import DurationPredictor


@patch("mlflow.pyfunc.load_model")
def test_duration_predictor(mock_load_model, tmp_path, monkeypatch):
    # Mock MLflow model loading to return a tuple or dummy model object
    model = LinearRegression()
    dv = DictVectorizer()
    dv.fit([{"PU_DO": "130_205", "trip_distance": 3.5}])
    mock_load_model.return_value = model

    dummy_model_path = tmp_path / "model.pkl"
    joblib.dump((model, dv), dummy_model_path)
    monkeypatch.setattr("prodml.config.settings.MODEL_PATH", dummy_model_path)

    predictor = DurationPredictor(model_path=str(dummy_model_path))

    # Bypass or mock the MLflow registry lookup if called inside load()
    with patch.object(predictor, "load"):
        predictor.model = model
        predictor.dv = dv

        sample_features = {"PU_DO": "130_205", "trip_distance": 3.5}
        pred = predictor.predict_one(sample_features)
        assert isinstance(pred, float)
        assert pred >= 0

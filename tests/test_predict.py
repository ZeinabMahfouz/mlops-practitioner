import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from prodml.predict import DurationPredictor


def test_duration_predictor(tmp_path, monkeypatch):
    # Create dummy model and vectorizer to match tuple unpacking (model, dv)
    dummy_model_path = tmp_path / "model.pkl"
    model = LinearRegression()
    dv = DictVectorizer()
    dv.fit([{"PU_DO": "130_205", "trip_distance": 3.5}])

    # Dump tuple as expected by DurationPredictor.load()
    joblib.dump((model, dv), dummy_model_path)

    # Point model path setting to the temporary file
    monkeypatch.setattr("prodml.config.settings.MODEL_PATH", dummy_model_path)

    predictor = DurationPredictor(model_path=str(dummy_model_path))
    predictor.load()

    sample_features = {"PU_DO": "130_205", "trip_distance": 3.5}

    pred = predictor.predict_one(sample_features)
    assert isinstance(pred, float)
    assert pred >= 0

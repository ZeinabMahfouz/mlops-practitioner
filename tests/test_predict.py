import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from prodml.predict import DurationPredictor


def test_duration_predictor(tmp_path, monkeypatch):
    # Create and fit dummy model and vectorizer so it's ready for prediction
    dummy_model_path = tmp_path / "model.pkl"
    model = LinearRegression()
    dv = DictVectorizer()

    # Fit the vectorizer and model with sample training data
    X_train = dv.fit_transform([{"PU_DO": "130_205", "trip_distance": 3.5}])
    y_train = [10.0]
    model.fit(X_train, y_train)

    # Save fitted model and vectorizer tuple
    joblib.dump((model, dv), dummy_model_path)

    # Point model path setting to temporary file
    monkeypatch.setattr("prodml.config.settings.MODEL_PATH", dummy_model_path)

    predictor = DurationPredictor(model_path=str(dummy_model_path))
    predictor.load()

    sample_features = {"PU_DO": "130_205", "trip_distance": 3.5}

    pred = predictor.predict_one(sample_features)
    assert isinstance(pred, float)
    assert pred >= 0

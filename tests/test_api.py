import sys
from pathlib import Path

import joblib
from fastapi.testclient import TestClient
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from prodml.api.app import app

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_health_check():
    # Test the health check endpoint response and status
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_predict_endpoint(tmp_path, monkeypatch):
    # Create a temporary dummy model file (model and vectorizer tuple) for API loading
    dummy_model_path = tmp_path / "model.pkl"
    model = LinearRegression()
    dv = DictVectorizer()
    dv.fit(
        [
            {
                "PULocationID": 130,
                "DOLocationID": 205,
                "trip_distance": 3.5,
                "PU_DO": "130_205",
                "duration": 10.0,
            }
        ]
    )
    joblib.dump((model, dv), dummy_model_path)

    # Override the application's model path setting to point to the temporary file
    monkeypatch.setattr("prodml.config.settings.MODEL_PATH", dummy_model_path)

    # Test the prediction endpoint with sample trip payload
    with TestClient(app) as client:
        payload = {
            "PULocationID": 130,
            "DOLocationID": 205,
            "trip_distance": 3.5,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], float)
        assert data["prediction"] >= 0

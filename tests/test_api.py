from fastapi.testclient import TestClient

from prodml.api.app import app


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_predict_endpoint():
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
        assert data["prediction"] > 0

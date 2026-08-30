import os
import pickle
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prodml.config import settings

# Set environment variables for MLflow tracking and MinIO S3 storage
os.environ["MLFLOW_TRACKING_URI"] = settings.MLFLOW_TRACKING_URI
os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT_URL

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


def timed(func: Callable) -> Callable:
    """Decorator to measure and log function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[{func.__name__}] executed in {end - start:.4f}s")
        return result

    return wrapper


class DurationPredictor:
    def __init__(
        self,
        dv_path: str = settings.MODEL_PATH,
        # Load by Alias instead of hardcoded version path to meet Step 03 requirements
        model_uri: str = "models:/ride-duration-xgboost@champion",
    ) -> None:
        self.dv_path = dv_path
        self.model_uri = model_uri
        self.dv = None
        self.model = None

    def load(self) -> None:
        """Load the DictVectorizer from local storage and the registered model via MLflow Registry URI."""
        # 1. Load local DictVectorizer artifact
        with open(self.dv_path, "rb") as f_in:
            self.dv, _ = pickle.load(f_in)

        # 2. Load the registered model using generic pyfunc representation
        self.model = mlflow.pyfunc.load_model(self.model_uri)

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        """Transform a single feature dictionary and return prediction."""
        X = self.dv.transform([features])
        preds = self.model.predict(X)
        return float(preds[0])

    def predict_batch(self, features_list: list[dict[str, Any]]) -> list[float]:
        """Transform a list of feature dictionaries and return batch predictions."""
        X = self.dv.transform(features_list)
        preds = self.model.predict(X)
        return [float(p) for p in preds]


# --- FastAPI Application Setup ---

app = FastAPI(title="Ride Duration Prediction API", version="1.0.0")
predictor = DurationPredictor()


class RideFeatures(BaseModel):
    PU_DO: str
    trip_distance: float


class PredictionResponse(BaseModel):
    duration_minutes: float
    model_version: str = "Champion"


@app.on_event("startup")
def startup_event():
    """Load model artifacts on application startup."""
    predictor.load()


@app.get("/health")
def health_check():
    """Health check endpoint returning model URI status."""
    return {"status": "ok", "model_uri": predictor.model_uri}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: RideFeatures):
    """Prediction endpoint for single ride duration inference."""
    if predictor.model is None or predictor.dv is None:
        raise HTTPException(status_code=500, detail="Model artifacts are not loaded.")

    try:
        payload = features.model_dump()
        prediction = predictor.predict_one(payload)
        return PredictionResponse(duration_minutes=round(prediction, 2))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))

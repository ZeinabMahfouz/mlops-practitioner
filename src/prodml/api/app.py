from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prodml.config import settings
from prodml.predict import DurationPredictor

predictor = DurationPredictor(model_path=str(settings.MODEL_PATH))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the model on startup
    try:
        predictor.load()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not load model on startup: {e}")
    yield


app = FastAPI(
    title="ProdML Duration Prediction API", version="0.1.0", lifespan=lifespan
)


class TripFeatures(BaseModel):
    PULocationID: int
    DOLocationID: int
    trip_distance: float


class PredictionOutput(BaseModel):
    prediction: float


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": predictor.model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(features: TripFeatures):
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    payload = {
        "PU_DO": f"{features.PULocationID}_{features.DOLocationID}",
        "trip_distance": features.trip_distance,
    }
    pred = predictor.predict_one(payload)
    return PredictionOutput(prediction=pred)

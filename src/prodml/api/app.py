import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prodml.config import settings
from prodml.predict import DurationPredictor

# Configure standard logging to comply with DoD (Zero print statements)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor = DurationPredictor(model_path=str(settings.MODEL_PATH))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the model on startup
    try:
        predictor.load()
        logger.info("Model loaded successfully on startup.")
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logger.warning(f"Could not load model on startup: {e}")
    yield


app = FastAPI(
    title="ProdML Duration Prediction API", version="0.1.0", lifespan=lifespan
)


# --- Request & Response Schemas ---


class TripFeatures(BaseModel):
    PULocationID: int
    DOLocationID: int
    trip_distance: float


class PredictionOutput(BaseModel):
    prediction: float


class BatchTripFeatures(BaseModel):
    trips: list[TripFeatures]


class BatchPredictionOutput(BaseModel):
    predictions: list[float]
    batch_size: int


class MetadataOutput(BaseModel):
    model_name: str
    model_version: str
    model_path: str
    framework: str
    model_loaded: bool


# --- API Endpoints ---


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": predictor.model is not None}


@app.get("/metadata", response_model=MetadataOutput)
def get_metadata():
    return MetadataOutput(
        model_name="NYC Green Taxi Duration Predictor",
        model_version="0.1.0",
        model_path=str(settings.MODEL_PATH),
        framework="scikit-learn / ONNX",
        model_loaded=predictor.model is not None,
    )


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


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(payload: BatchTripFeatures):
    if predictor.model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    formatted_payloads = [
        {
            "PU_DO": f"{trip.PULocationID}_{trip.DOLocationID}",
            "trip_distance": trip.trip_distance,
        }
        for trip in payload.trips
    ]

    # Batch prediction execution
    preds = [predictor.predict_one(item) for item in formatted_payloads]

    return BatchPredictionOutput(
        predictions=preds,
        batch_size=len(preds),
    )

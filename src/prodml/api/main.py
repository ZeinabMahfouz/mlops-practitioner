import mlflow
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction import DictVectorizer

from prodml.config import settings

app = FastAPI(title="Ride Duration Prediction API")


class TripInput(BaseModel):
    PU_DO: str
    trip_distance: float


@app.post("/predict")
def predict(trip: TripInput):
    # تعيين رابط التتبع الخاص بـ MLflow للتأكد من الاتصال بالخير الخارجي
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URL)

    model_uri = "models:/ride-duration-predictor/latest"

    model = mlflow.xgboost.load_model(model_uri)

    input_data = [{"PU_DO": trip.PU_DO, "trip_distance": trip.trip_distance}]
    dv = DictVectorizer(sparse=True)
    X = dv.fit_transform(input_data)

    dmatrix = xgb.DMatrix(X)
    prediction = model.predict(dmatrix)

    return {"predicted_duration_minutes": float(prediction[0])}

import numpy as np
import xgboost as xgb

import bentoml
from bentoml.models import BentoModel


@bentoml.service(workers="cpu_count")
class RideDurationModel:
    bento_model = BentoModel("ride_duration_xgb:latest")

    def __init__(self):
        self.booster: xgb.Booster = self.bento_model.load_model()

    @bentoml.api(batchable=True, batch_dim=0, max_batch_size=32, max_latency_ms=500)
    def predict(self, features: np.ndarray) -> np.ndarray:
        dmatrix = xgb.DMatrix(features)
        return self.booster.predict(dmatrix)


@bentoml.service(workers="cpu_count")
class RideDurationService:
    model = bentoml.depends(RideDurationModel)
    bento_model = BentoModel("ride_duration_xgb:latest")

    def __init__(self):
        self.dv = self.bento_model.custom_objects["dv"]

    @bentoml.api
    async def predict(
        self, PULocationID: int, DOLocationID: int, trip_distance: float
    ) -> dict:
        features = {
            "PU_DO": f"{PULocationID}_{DOLocationID}",
            "trip_distance": trip_distance,
        }
        X = self.dv.transform([features])
        X_dense = X.toarray().astype(np.float32) if hasattr(X, "toarray") else X
        pred = await self.model.to_async.predict(X_dense)
        return {"prediction": float(pred[0])}

    @bentoml.api
    async def predict_batch(self, trips: list[dict]) -> dict:
        dicts = [
            {
                "PU_DO": f"{t['PULocationID']}_{t['DOLocationID']}",
                "trip_distance": t["trip_distance"],
            }
            for t in trips
        ]
        X = self.dv.transform(dicts)
        X_dense = X.toarray().astype(np.float32) if hasattr(X, "toarray") else X
        preds = await self.model.to_async.predict(X_dense)
        return {"predictions": [float(p) for p in preds], "batch_size": len(preds)}

    @bentoml.api
    def metadata(self) -> dict:
        return {
            "model_tag": str(self.bento_model.tag),
            "framework": "xgboost",
            "runner_workers": "cpu_count",
        }

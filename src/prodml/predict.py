import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


def timed(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info("[%s] executed in %.4fs", func.__name__, end - start)
        return result

    return wrapper


class DurationPredictor:
    def __init__(self, model_uri: str = "models:/ride-duration-predictor/Production") -> None:
        self.model_uri = model_uri
        self.model = None

    def load(self) -> None:
        self.model = mlflow.pyfunc.load_model(self.model_uri)

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        preds = self.model.predict([features])
        return float(preds[0])

    def predict_batch(self, features_list: list[dict[str, Any]]) -> list[float]:
        preds = self.model.predict(features_list)
        return [float(p) for p in preds]
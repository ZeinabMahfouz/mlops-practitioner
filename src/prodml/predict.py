import pickle
import time
from collections.abc import Callable
from functools import wraps
from typing import Any


def timed(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[{func.__name__}] executed in {end - start:.4f}s")
        return result

    return wrapper


class DurationPredictor:
    def __init__(self, model_path: str = "models/model.pkl") -> None:
        self.model_path = model_path
        self.dv = None
        self.model = None

    def load(self) -> None:
        with open(self.model_path, "rb") as f_in:
            self.dv, self.model = pickle.load(f_in)

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        X = self.dv.transform([features])
        preds = self.model.predict(X)
        return float(preds[0])

    def predict_batch(self, features_list: list[dict[str, Any]]) -> list[float]:
        X = self.dv.transform(features_list)
        preds = self.model.predict(X)
        return [float(p) for p in preds]

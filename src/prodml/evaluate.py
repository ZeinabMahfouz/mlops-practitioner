import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import engineer_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model():
    logger.info("Starting model evaluation...")

    # 1. تحميل البيانات وتطبيق الفلترة والخصائص
    df_raw = load_data(str(settings.DATA_PATH))
    df = engineer_features(df_raw)
    _, df_val = split_data(df)

    # 2. تحميل (DictVectorizer + Model) المخزنة في train.py
    with open(settings.MODEL_PATH, "rb") as f_in:
        dv, model = pickle.load(f_in)

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    dicts_val = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(dicts_val)
    y_val = df_val["duration"].values

    # 3. التنبؤ وحساب المقاييس
    y_pred = model.predict(X_val)
    mae = float(mean_absolute_error(y_val, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

    logger.info(f"Evaluation Results -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    # 4. حفظ المقاييس في reports/metrics.json ليتعرف عليها DVC
    metrics_path = Path("reports/metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {"mae": mae, "rmse": rmse}

    with open(metrics_path, "w") as f_out:
        json.dump(metrics, f_out, indent=4)

    logger.info(f"Metrics successfully saved to {metrics_path}")


if __name__ == "__main__":
    evaluate_model()

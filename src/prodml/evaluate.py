import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model, dv, test_df):
    """Calculate evaluation metrics for the model on test data"""
    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    dicts_test = test_df[categorical + numerical].to_dict(orient="records")
    X_test = dv.transform(dicts_test)
    y_test = test_df["duration"].values

    test_dmatrix = xgb.DMatrix(X_test)
    y_pred = model.predict(test_dmatrix)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    return {"test_mae": float(mae), "test_rmse": rmse}


def main():
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/model.pkl")
    input_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/features")
    reports_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1.load the saved model and DictVectorizer
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    dv = artifact["dv"]

    # 2.load test data
    test_df = pd.read_parquet(input_dir / "test.parquet")

    # 3.calculate evaluation metrics
    metrics = evaluate_model(model, dv, test_df)

    # 4. save the metrics to a JSON file in the reports directory
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(
        f"Evaluation metrics saved to {metrics_path} | MAE: {metrics['test_mae']:.4f}, RMSE: {metrics['test_rmse']:.4f}"
    )


def evaluate_and_log(
    model_path: Path,
    input_dir: Path,
    reports_dir: Path,
    run_id: str | None = None,
) -> dict:
    model_path = Path(model_path)
    input_dir = Path(input_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    dv = artifact["dv"]

    test_df = pd.read_parquet(input_dir / "test.parquet")
    metrics = evaluate_model(model, dv, test_df)

    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    if run_id is not None:
        import mlflow
        from mlflow.tracking import MlflowClient

        from prodml.config import settings

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URL)
        client = MlflowClient()
        client.log_metric(run_id, "held_out_test_mae", metrics["test_mae"])
        client.log_metric(run_id, "held_out_test_rmse", metrics["test_rmse"])

    logger.info(
        "Evaluation metrics saved to %s | MAE: %.4f, RMSE: %.4f",
        metrics_path,
        metrics["test_mae"],
        metrics["test_rmse"],
    )
    return metrics


if __name__ == "__main__":
    main()

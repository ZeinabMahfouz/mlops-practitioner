import logging
import os
import pickle
import subprocess
import time

import boto3
import mlflow
import mlflow.onnx
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import xgboost as xgb
from botocore.exceptions import ClientError
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import engineer_features

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure MLflow and MinIO environment variables
os.environ["MLFLOW_TRACKING_URI"] = settings.MLFLOW_TRACKING_URI
os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = settings.MLFLOW_S3_ENDPOINT_URL

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)


def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def prepare_dataset():
    """Load data, engineer features, and vectorize inputs."""
    logger.info("Loading and preparing dataset...")
    df_raw = load_data(str(settings.DATA_PATH))
    df = engineer_features(df_raw)
    df_train, df_val = split_data(df)

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    dicts_train = df_train[categorical + numerical].to_dict(orient="records")
    dicts_val = df_val[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts_train)
    X_val = dv.transform(dicts_val)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    return dv, X_train, X_val, y_train, y_val


def ensure_minio_bucket(bucket_name: str = "mlflow"):
    """Ensure the target MinIO/S3 bucket exists before MLflow uploads artifacts."""
    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.MLFLOW_S3_ENDPOINT_URL,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError:
        logger.info(f"Bucket '{bucket_name}' not found. Creating it...")
        s3_client.create_bucket(Bucket=bucket_name)


def train_ridge_baseline(dv, X_train, X_val, y_train, y_val, git_hash: str):
    """Train Ridge regression baseline model and log to MLflow."""
    logger.info("Starting Ridge Baseline Run...")
    with mlflow.start_run(run_name="Ridge_Baseline"):
        start_time = time.time()
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        duration = time.time() - start_time

        y_pred = model.predict(X_val)
        mae = float(mean_absolute_error(y_val, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

        logger.info(f"Ridge Validation MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        # Log params, metrics, tags
        mlflow.log_params({"alpha": 1.0, "model_type": "Ridge"})
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "train_duration_sec": duration})
        mlflow.set_tags(
            {
                "git_commit": git_hash,
                "framework": "scikit-learn",
                "data_path": str(settings.DATA_PATH),
            }
        )

        # Save local pickle
        settings.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(settings.MODEL_PATH, "wb") as f_out:
            pickle.dump((dv, model), f_out)
        logger.info(f"Model saved locally to {settings.MODEL_PATH}")

        # Log sklearn model to MLflow/MinIO
        mlflow.sklearn.log_model(model, name="model")

        # Export and Log ONNX Model
        num_features = X_train.shape[1]
        initial_type = [("float_input", FloatTensorType([None, num_features]))]
        try:
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            onnx_path = settings.MODEL_PATH.with_suffix(".onnx")
            with open(onnx_path, "wb") as f_onnx:
                f_onnx.write(onnx_model.SerializeToString())
            mlflow.log_artifact(str(onnx_path), artifact_path="onnx")
            logger.info(f"ONNX model saved to {onnx_path} and logged to MLflow")
        except (RuntimeError, ValueError, TypeError) as e:
            logger.warning(f"Could not export ONNX model: {e}")


def run_xgboost_sweep(
    X_train, X_val, y_train, y_val, git_hash: str, n_trials: int = 10
):
    """Perform hyperparameter optimization sweep for XGBoost with Optuna."""
    logger.info("Starting XGBoost Optuna Hyperparameter Sweep...")

    with mlflow.start_run(run_name="XGBoost_Optuna_Parent"):
        mlflow.set_tag("experiment_type", "optuna_sweep")

        def objective(trial):
            with mlflow.start_run(run_name=f"XGB_Trial_{trial.number}", nested=True):
                params = {
                    "max_depth": trial.suggest_int("max_depth", 3, 9),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                    "random_state": 42,
                }

                start_time = time.time()
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train)
                duration = time.time() - start_time

                y_pred = model.predict(X_val)
                mae = float(mean_absolute_error(y_val, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

                mlflow.log_params(params)
                mlflow.log_metrics(
                    {"mae": mae, "rmse": rmse, "train_duration_sec": duration}
                )
                mlflow.set_tags(
                    {
                        "git_commit": git_hash,
                        "framework": "xgboost",
                        "trial": trial.number,
                    }
                )
                mlflow.xgboost.log_model(model, name="model")
                return mae

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_mae", study.best_value)
        logger.info(f"XGBoost Sweep Complete! Best MAE: {study.best_value:.4f}")


def main() -> None:
    ensure_minio_bucket("mlflow")  # Create bucket if missing
    git_hash = get_git_commit()
    dv, X_train, X_val, y_train, y_val = prepare_dataset()

    # 1. Ridge Baseline Run
    train_ridge_baseline(dv, X_train, X_val, y_train, y_val, git_hash)

    # 2. XGBoost Optuna Sweep
    run_xgboost_sweep(X_train, X_val, y_train, y_val, git_hash, n_trials=10)


if __name__ == "__main__":
    main()

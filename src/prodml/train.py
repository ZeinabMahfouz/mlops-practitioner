import logging
import pickle
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_bucket_exists_for_uri(artifact_uri):
    """Ensure the S3/MinIO bucket specified in the MLflow artifact URI exists."""
    parsed = urlparse(artifact_uri)
    if parsed.scheme == "s3":
        bucket_name = parsed.netloc
        s3 = boto3.client(
            "s3",
            endpoint_url=settings.MLFLOW_S3_ENDPOINT_URL or "http://localhost:9000",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        try:
            s3.head_bucket(Bucket=bucket_name)
        except Exception:  # noqa: BLE001
            try:
                s3.create_bucket(Bucket=bucket_name)
                logger.info(f"Bucket '{bucket_name}' created automatically in MinIO.")
            except Exception as e:  # noqa: BLE001
                logger.error(f"Could not create bucket {bucket_name}: {e}")


def load_features(input_dir: Path):
    """Load train and test parquet features and vectorize them."""
    import pandas as pd
    train_df = pd.read_parquet(input_dir / "train.parquet")
    val_df = pd.read_parquet(input_dir / "test.parquet")

    categorical = ["PU_DO"]
    numerical = ["trip_distance"]

    dicts_train = train_df[categorical + numerical].to_dict(orient="records")
    dicts_val = val_df[categorical + numerical].to_dict(orient="records")

    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts_train)
    X_val = dv.transform(dicts_val)

    y_train = train_df["duration"].values
    y_val = val_df["duration"].values

    return X_train, X_val, y_train, y_val, dv


def train_xgboost(X_train, X_val, y_train, y_val, params: dict):
    """Train an XGBoost regression model with MLflow tracking."""
    mlflow.xgboost.autolog(disable=True)
    with mlflow.start_run(run_name="xgboost_model", nested=True) as run:
        ensure_bucket_exists_for_uri(run.info.artifact_uri)
        
        mlflow.log_params(params)
        mlflow.set_tag("git_commit", "v0.1.0")
        mlflow.set_tag("data_version", "2024-01")
        mlflow.set_tag("framework", "xgboost")

        train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        val_dmatrix = xgb.DMatrix(X_val, label=y_val)

        evals = [(val_dmatrix, "validation")]
        model = xgb.train(
            params,
            train_dmatrix,
            num_boost_round=100,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=False,
        )

        y_pred = model.predict(val_dmatrix)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        mlflow.xgboost.log_model(
            model, 
            "model",
            registered_model_name="ride-duration-predictor"
        )
        return model, mae


def main() -> None:
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/features")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URL)
    mlflow.set_experiment("ride-duration-prediction")

    X_train, X_val, y_train, y_val, dv = load_features(input_dir)

    params = {
        "learning_rate": 0.1,
        "max_depth": 6,
        "objective": "reg:squarederror",
        "eval_metric": "mae"
    }

    with mlflow.start_run(run_name="xgboost_sweep") as parent_run:
        ensure_bucket_exists_for_uri(parent_run.info.artifact_uri)
        model, mae = train_xgboost(X_train, X_val, y_train, y_val, params)

    # حفظ النموذج والـ DictVectorizer معاً في ملف واحد لكي تستخدمه مرحلة evaluate والتنبؤ لاحقاً
    model_artifact = {
        "model": model,
        "dv": dv
    }
    model_path = output_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_artifact, f)
    
    logger.info(f"Model successfully saved to {model_path} with MAE: {mae:.4f}")


if __name__ == "__main__":
    main()
import argparse
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
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import settings

EXPERIMENT_NAME = "ride-duration-prediction"

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


def train_xgboost(
    X_train, X_val, y_train, y_val, params: dict, logical_date: str | None = None
):
    mlflow.xgboost.autolog(disable=True)
    with mlflow.start_run(run_name="xgboost_model", nested=True) as run:
        ensure_bucket_exists_for_uri(run.info.artifact_uri)

        mlflow.log_params(params)
        mlflow.set_tag("git_commit", "v0.1.0")
        mlflow.set_tag("data_version", "2024-01")
        mlflow.set_tag("framework", "xgboost")
        if logical_date is not None:
            mlflow.set_tag("logical_date", logical_date)

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
            model, "model", registered_model_name="ride-duration-predictor"
        )
        return model, {"mae": mae, "rmse": rmse, "run_id": run.info.run_id}


def parse_args(argv=None):
    """Parse CLI args explicitly so pytest's own flags (--cov, -v, etc.)
    never get misread as positional input/output paths."""
    parser = argparse.ArgumentParser(description="Train the ride-duration model.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="data/features",
        type=Path,
        help="Directory containing train.parquet / test.parquet",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="models",
        type=Path,
        help="Directory to write the trained model artifact to",
    )
    # parse_known_args ignores any extra flags (e.g. pytest's) instead of erroring
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv=None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URL)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_val, y_train, y_val, dv = load_features(input_dir)

    params = {
        "learning_rate": 0.1,
        "max_depth": 6,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
    }

    with mlflow.start_run(run_name="xgboost_sweep") as parent_run:
        ensure_bucket_exists_for_uri(parent_run.info.artifact_uri)
        model, metrics = train_xgboost(X_train, X_val, y_train, y_val, params)

    # Save the model and DictVectorizer together as a single artifact for later evaluation
    model_artifact = {"model": model, "dv": dv}
    model_path = output_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_artifact, f)

    logger.info(
        f"Model successfully saved to {model_path} with MAE: {metrics['mae']:.4f}"
    )


def run_training(
    input_dir: Path, output_dir: Path, logical_date: str | None = None
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URL)
    mlflow.set_experiment(EXPERIMENT_NAME)

    model_suffix = f"_{logical_date}" if logical_date else ""
    model_path = output_dir / f"model{model_suffix}.pkl"

    if logical_date is not None:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is not None:
            existing = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=(
                    f"tags.logical_date = '{logical_date}' and "
                    "tags.mlflow.runName = 'xgboost_model'"
                ),
                order_by=["start_time DESC"],
                max_results=1,
            )
            if existing and model_path.exists():
                run = existing[0]
                logger.info(
                    "Found existing run %s for logical_date=%s -- skipping "
                    "re-training (idempotency guard).",
                    run.info.run_id,
                    logical_date,
                )
                return {
                    "run_id": run.info.run_id,
                    "model_path": str(model_path),
                    "mae": run.data.metrics.get("mae"),
                    "rmse": run.data.metrics.get("rmse"),
                    "reused": True,
                }

    X_train, X_val, y_train, y_val, dv = load_features(input_dir)
    params = {
        "learning_rate": 0.1,
        "max_depth": 6,
        "objective": "reg:squarederror",
        "eval_metric": "mae",
    }

    with mlflow.start_run(run_name="xgboost_sweep") as parent_run:
        ensure_bucket_exists_for_uri(parent_run.info.artifact_uri)
        model, metrics = train_xgboost(
            X_train, X_val, y_train, y_val, params, logical_date=logical_date
        )

    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "dv": dv}, f)

    logger.info(
        "Model trained for logical_date=%s -- run %s, MAE %.4f, RMSE %.4f, saved to %s",
        logical_date,
        metrics["run_id"],
        metrics["mae"],
        metrics["rmse"],
        model_path,
    )
    return {
        "run_id": metrics["run_id"],
        "model_path": str(model_path),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "reused": False,
    }


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.evaluate import evaluate_and_log
from prodml.features import engineer_features
from prodml.registry import promote_best_model
from prodml.train import run_training
from prodml.validate import validate_features

logger = logging.getLogger(__name__)

RAW_DATA_PATH = Path("/opt/airflow/data/raw/green_tripdata_2024-01.parquet")
FEATURES_DIR = Path("/opt/airflow/data/features")
MODELS_DIR = Path("/opt/airflow/models")
REPORTS_DIR = Path("/opt/airflow/reports")

EXPERIMENT_NAME = "ride-duration-prediction"
MODEL_NAME = "ride-duration-predictor"


def decide_promotion(run_id: str, reports_dir: Path = REPORTS_DIR) -> str:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URL)
    client = MlflowClient()

    run = client.get_run(run_id)
    new_rmse = run.data.metrics.get("rmse")

    staging_versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if staging_versions:
        current_best = client.get_run(staging_versions[0].run_id).data.metrics.get(
            "rmse"
        )
        source = "current Staging model"
    else:
        current_best = None
        baseline_path = Path(reports_dir) / "metrics.json"
        if baseline_path.exists():
            current_best = json.loads(baseline_path.read_text()).get("test_rmse")
        source = "Module 1 baseline (reports/metrics.json)"

    logger.info(
        "decide_promotion: new run rmse=%s vs %s (%s)", new_rmse, current_best, source
    )

    if new_rmse is None:
        logger.warning("Run %s has no rmse metric -- skipping promotion.", run_id)
        return "skip_registration"

    if current_best is None or new_rmse < current_best:
        return "register_model"
    return "skip_registration"


def register(run_id: str) -> None:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URL)
    client = MlflowClient()

    existing = client.search_model_versions(f"run_id='{run_id}'")
    if existing:
        logger.info(
            "Run %s is already registered as version %s -- skipping duplicate "
            "registration.",
            run_id,
            existing[0].version,
        )
        return

    promote_best_model()


def skip(run_id: str) -> None:
    logger.info(
        "Run %s did not improve on the current best model -- not promoting.", run_id
    )


def extract_task(**context) -> str:
    df = load_data(str(RAW_DATA_PATH))
    train_df, test_df = split_data(df)
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(FEATURES_DIR / "train.parquet")
    test_df.to_parquet(FEATURES_DIR / "test.parquet")
    logger.info(
        "extract: %d train / %d test rows -> %s",
        len(train_df),
        len(test_df),
        FEATURES_DIR,
    )
    return str(FEATURES_DIR)


def validate_task(**context) -> str:
    features_dir = context["ti"].xcom_pull(task_ids="extract")
    validate_features(Path(features_dir))
    return features_dir


def train_task(**context) -> dict:
    features_dir = context["ti"].xcom_pull(task_ids="validate")
    logical_date = context["ds"]  # Airflow's logical date, YYYY-MM-DD
    return run_training(Path(features_dir), MODELS_DIR, logical_date=logical_date)


def evaluate_task(**context) -> dict:
    train_result = context["ti"].xcom_pull(task_ids="train")
    features_dir = context["ti"].xcom_pull(task_ids="validate")
    metrics = evaluate_and_log(
        model_path=Path(train_result["model_path"]),
        input_dir=Path(features_dir),
        reports_dir=REPORTS_DIR,
        run_id=train_result["run_id"],
    )
    return {**train_result, **metrics}


def decide_promotion_task(**context) -> str:
    train_result = context["ti"].xcom_pull(task_ids="train")
    return decide_promotion(train_result["run_id"], reports_dir=REPORTS_DIR)


def register_task(**context) -> None:
    train_result = context["ti"].xcom_pull(task_ids="train")
    register(train_result["run_id"])


def skip_task(**context) -> None:
    train_result = context["ti"].xcom_pull(task_ids="train")
    skip(train_result["run_id"])


def notify_task(**context) -> None:
    train_result = context["ti"].xcom_pull(task_ids="train")
    branch = context["ti"].xcom_pull(task_ids="decide_promotion")
    logger.info(
        "Training DAG finished for logical_date=%s: run=%s mae=%.4f rmse=%.4f "
        "branch=%s reused=%s",
        context["ds"],
        train_result["run_id"],
        train_result["mae"],
        train_result["rmse"],
        branch,
        train_result["reused"],
    )

import json
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from prodml.config import settings
from prodml.registry import promote_best_model

EXPERIMENT_NAME = "ride-duration-prediction"
MODEL_NAME = "ride-duration-predictor"
TRAINING_RUN_NAME = "xgboost_model"  # nested run inside train.py that logs rmse
BASELINE_METRICS_PATH = Path("reports/metrics.json")


def get_latest_training_run(client: MlflowClient):
    """Get the most recent nested run that actually logged an rmse metric."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found.")
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{TRAINING_RUN_NAME}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def get_current_staging_rmse(client: MlflowClient):
    """Return the RMSE of whatever model is currently in Staging, if any."""
    versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not versions:
        return None

    staging_version = versions[0]
    run = client.get_run(staging_version.run_id)
    return run.data.metrics.get("rmse")


def get_baseline_rmse():
    """Fallback threshold from the Module 1 baseline evaluation."""
    if not BASELINE_METRICS_PATH.exists():
        print(f"No baseline metrics found at {BASELINE_METRICS_PATH}.")
        return None
    with open(BASELINE_METRICS_PATH) as f:
        metrics = json.load(f)
    return metrics.get("test_rmse")


def main():
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URL)
    client = MlflowClient()

    latest_run = get_latest_training_run(client)
    if latest_run is None:
        print("No training run found to gate-check. Failing.")
        sys.exit(1)

    new_rmse = latest_run.data.metrics.get("rmse")
    if new_rmse is None:
        print(f"Run {latest_run.info.run_id} has no rmse metric logged. Failing.")
        sys.exit(1)

    current_best = get_current_staging_rmse(client)
    source = "current Staging model"
    if current_best is None:
        print("No model currently in Staging - falling back to baseline metrics.")
        current_best = get_baseline_rmse()
        source = "baseline (reports/metrics.json)"

    if current_best is None:
        print("No baseline available either. Failing gate check.")
        sys.exit(1)

    print(f"New run RMSE:      {new_rmse:.4f}")
    print(f"Comparing against: {current_best:.4f} ({source})")

    if new_rmse < current_best:
        print("PASSED gate check - new model improves RMSE. Promoting to Staging.")
        promote_best_model()
        sys.exit(0)
    else:
        print(
            f"FAILED gate check - new RMSE ({new_rmse:.4f}) did not improve on "
            f"{source} ({current_best:.4f}). Not promoting."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from prodml.config import settings

mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
client = MlflowClient()


def promote_if_better(
    candidate_run_id: str,
    metric: str = "mae",
    model_name: str = "ride-duration-xgboost",
) -> bool:
    """Promote candidate model to @champion if its metric beats the current champion."""
    # 1. Fetch candidate run metric
    candidate_run = client.get_run(candidate_run_id)
    candidate_metric = candidate_run.data.metrics.get(metric)

    if candidate_metric is None:
        print(f"Metric '{metric}' not found in candidate run {candidate_run_id}.")
        return False

    # 2. Try fetching current champion model metric
    try:
        champion_version = client.get_model_version_by_alias(model_name, "champion")
        champion_run = client.get_run(champion_version.run_id)
        champion_metric = champion_run.data.metrics.get(metric)
    except MlflowException:
        champion_metric = float("inf")  # If no champion exists yet

    print(
        f"Candidate {metric}: {candidate_metric:.4f} | Current Champion {metric}: {champion_metric:.4f}"
    )

    # 3. Promote only if candidate is strictly better (lower is better for MAE/RMSE)
    if candidate_metric < champion_metric:
        print(f"Promoting run {candidate_run_id} to @champion...")
        model_version = mlflow.register_model(
            f"runs:/{candidate_run_id}/model", model_name
        )
        client.set_registered_model_alias(model_name, "champion", model_version.version)
        return True

    print("Candidate did not outperform the current champion.")
    return False

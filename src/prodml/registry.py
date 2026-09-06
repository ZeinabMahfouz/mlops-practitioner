import mlflow
from mlflow.tracking import MlflowClient
from prodml.config import settings

def promote_best_model():
    # Set tracking URI
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URL)
    client = MlflowClient()
    
    experiment_name = "ride-duration-prediction"
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    
    # Search for all runs in the experiment and sort by lowest RMSE
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )
    
    if not runs:
        print("No runs found in the experiment.")
        return
    
    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_rmse = best_run.data.metrics.get("rmse")
    print(f"Best run found: {best_run_id} with RMSE: {best_rmse:.4f}")
    
    # Model name to register under
    model_name = "ride-duration-predictor"
    artifact_path = "model"
    model_uri = f"runs:/{best_run_id}/{artifact_path}"
    
    # Register the model
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Model registered as '{model_name}', version {model_version.version}")
    
    # Transition model version to Staging
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=False
    )
    print(f"Model version {model_version.version} successfully transitioned to Staging.")

if __name__ == "__main__":
    promote_best_model()
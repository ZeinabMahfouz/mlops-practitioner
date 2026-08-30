from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATA_PATH: Path = Path("data/raw/green_tripdata_2024-01.parquet")
    MODEL_PATH: Path = Path("models/model.pkl")
    PORT: int = 8000

    # MLflow & MinIO Tracking Settings
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "ride-duration-prediction"
    AWS_ACCESS_KEY_ID: str = "minioadmin"
    AWS_SECRET_ACCESS_KEY: str = "minioadminpassword"
    MLFLOW_S3_ENDPOINT_URL: str = "http://localhost:9000"


settings = Settings()

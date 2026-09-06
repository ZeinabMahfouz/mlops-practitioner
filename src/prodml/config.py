import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATA_PATH: Path = Path("data/green_tripdata_2024-01.parquet")
    MODEL_PATH: Path = Path("models/model.pkl")
    PORT: int = 8000
    
    MLFLOW_TRACKING_URL: str = "http://localhost:5000"
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    MLFLOW_S3_ENDPOINT_URL: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # حقن المتغيرات تلقائياً في البيئة لتراها مكتبة boto3
        if self.AWS_ACCESS_KEY_ID:
            os.environ["AWS_ACCESS_KEY_ID"] = self.AWS_ACCESS_KEY_ID
        if self.AWS_SECRET_ACCESS_KEY:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.AWS_SECRET_ACCESS_KEY
        if self.MLFLOW_S3_ENDPOINT_URL:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.MLFLOW_S3_ENDPOINT_URL
        if self.MLFLOW_TRACKING_URL:
            os.environ["MLFLOW_TRACKING_URL"] = self.MLFLOW_TRACKING_URL


settings = Settings()
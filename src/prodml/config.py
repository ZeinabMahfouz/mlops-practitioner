from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATA_PATH: Path = Path("data/green_tripdata_2024-01.parquet")
    MODEL_PATH: Path = Path("models/model.pkl")
    PORT: int = 8000


settings = Settings()

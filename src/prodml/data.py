import os
import pandas as pd
import yaml


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_parquet(data_path)


def split_data(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_size = int(len(df) * train_ratio)
    return df.iloc[:train_size], df.iloc[train_size:]


if __name__ == "__main__":
    # 1. قراءة المسارات من params.yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    raw_path = params["prepare"]["raw_data_path"]
    prepared_path = params["prepare"]["prepared_data_path"]

    # create the directory for the prepared data if it doesn't exist
    os.makedirs(os.path.dirname(prepared_path), exist_ok=True)

    # read the raw data, prepare it, and save it to the prepared path
    df = load_data(raw_path)
    df.to_parquet(prepared_path)
    print(f"Data prepared successfully: {len(df)} rows saved to {prepared_path}")

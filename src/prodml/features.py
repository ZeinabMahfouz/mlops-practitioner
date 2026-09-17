import sys
from pathlib import Path

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["duration"] = (
        df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    ).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df = df[(df.trip_distance > 0) & (df.trip_distance <= 100)].copy()

    df["PULocationID"] = df["PULocationID"].fillna(-1).astype(int)
    df["DOLocationID"] = df["DOLocationID"].fillna(-1).astype(int)
    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    return df


if __name__ == "__main__":
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    # read the input data
    train_df = pd.read_parquet(input_dir / "train.parquet")
    test_df = pd.read_parquet(input_dir / "test.parquet")

    # feature engineering
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)

    # save the processed data to the output directory
    train_df.to_parquet(output_dir / "train.parquet")
    test_df.to_parquet(output_dir / "test.parquet")

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

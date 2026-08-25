import pandas as pd


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_parquet(data_path)


def split_data(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_size = int(len(df) * train_ratio)
    return df.iloc[:train_size], df.iloc[train_size:]

import sys
from pathlib import Path

import pandas as pd


def load_data(data_path: str) -> pd.DataFrame:
    return pd.read_parquet(data_path)


def split_data(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_size = int(len(df) * train_ratio)
    return df.iloc[:train_size], df.iloc[train_size:]


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    train_df, test_df = split_data(df)

    train_df.to_parquet(output_dir / "train.parquet")
    test_df.to_parquet(output_dir / "test.parquet")
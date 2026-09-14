from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"PU_DO", "trip_distance", "duration"}
MIN_ROWS = 100


class DataValidationError(ValueError):
    """Raised when the extracted features fail a validation check."""


def validate_features(features_dir: Path) -> None:
    features_dir = Path(features_dir)

    for split in ("train", "test"):
        path = features_dir / f"{split}.parquet"
        if not path.exists():
            raise DataValidationError(f"{path} does not exist.")

        df = pd.read_parquet(path)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise DataValidationError(f"{path} is missing columns: {missing}")

        if len(df) < MIN_ROWS:
            raise DataValidationError(
                f"{path} has only {len(df)} rows, expected at least {MIN_ROWS}."
            )

        if df["duration"].isna().any():
            raise DataValidationError(f"{path} has null values in 'duration'.")

        if not df["trip_distance"].between(0, 100).all():
            raise DataValidationError(
                f"{path} has trip_distance values outside the expected [0, 100] range."
            )

        if not df["duration"].between(1, 60).all():
            raise DataValidationError(
                f"{path} has duration values outside the expected [1, 60] range."
            )

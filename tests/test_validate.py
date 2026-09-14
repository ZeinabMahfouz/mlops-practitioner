import pandas as pd
import pytest

from prodml.validate import DataValidationError, validate_features


def _write_valid_split(tmp_path, split: str, n: int = 150):
    df = pd.DataFrame(
        {
            "PU_DO": [f"{i}_{i + 1}" for i in range(n)],
            "trip_distance": [1.0 + (i % 10) for i in range(n)],
            "duration": [5.0 + (i % 20) for i in range(n)],
        }
    )
    df.to_parquet(tmp_path / f"{split}.parquet")


def test_validate_features_passes_on_valid_data(tmp_path):
    _write_valid_split(tmp_path, "train")
    _write_valid_split(tmp_path, "test")

    validate_features(tmp_path)  # should not raise


def test_validate_features_missing_file(tmp_path):
    _write_valid_split(tmp_path, "train")
    # no test.parquet written

    with pytest.raises(DataValidationError, match="does not exist"):
        validate_features(tmp_path)


def test_validate_features_missing_column(tmp_path):
    df = pd.DataFrame({"trip_distance": [1.0] * 150, "duration": [5.0] * 150})
    df.to_parquet(tmp_path / "train.parquet")
    _write_valid_split(tmp_path, "test")

    with pytest.raises(DataValidationError, match="missing columns"):
        validate_features(tmp_path)


def test_validate_features_too_few_rows(tmp_path):
    _write_valid_split(tmp_path, "train", n=5)
    _write_valid_split(tmp_path, "test")

    with pytest.raises(DataValidationError, match="only 5 rows"):
        validate_features(tmp_path)


def test_validate_features_null_duration(tmp_path):
    n = 150
    df = pd.DataFrame(
        {
            "PU_DO": [f"{i}_{i + 1}" for i in range(n)],
            "trip_distance": [1.0] * n,
            "duration": [5.0] * (n - 1) + [None],
        }
    )
    df.to_parquet(tmp_path / "train.parquet")
    _write_valid_split(tmp_path, "test")

    with pytest.raises(DataValidationError, match="null values"):
        validate_features(tmp_path)


def test_validate_features_out_of_range_distance(tmp_path):
    n = 150
    df = pd.DataFrame(
        {
            "PU_DO": [f"{i}_{i + 1}" for i in range(n)],
            "trip_distance": [1.0] * (n - 1) + [500.0],
            "duration": [5.0] * n,
        }
    )
    df.to_parquet(tmp_path / "train.parquet")
    _write_valid_split(tmp_path, "test")

    with pytest.raises(DataValidationError, match="trip_distance"):
        validate_features(tmp_path)


def test_validate_features_out_of_range_duration(tmp_path):
    n = 150
    df = pd.DataFrame(
        {
            "PU_DO": [f"{i}_{i + 1}" for i in range(n)],
            "trip_distance": [1.0] * n,
            "duration": [5.0] * (n - 1) + [120.0],
        }
    )
    df.to_parquet(tmp_path / "train.parquet")
    _write_valid_split(tmp_path, "test")

    with pytest.raises(DataValidationError, match="duration"):
        validate_features(tmp_path)

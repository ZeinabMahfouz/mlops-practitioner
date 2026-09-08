import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from unittest.mock import patch

from prodml.data import split_data
from prodml.features import engineer_features
from prodml.train import main as train_main


def test_engineer_features():
    df = pd.DataFrame(
        {
            "PULocationID": [10, 20],
            "DOLocationID": [15, 25],
            "trip_distance": [1.2, 3.4],
            "lpep_pickup_datetime": pd.to_datetime(
                ["2023-01-01 10:00:00", "2023-01-01 11:00:00"]
            ),
            "lpep_dropoff_datetime": pd.to_datetime(
                ["2023-01-01 10:15:00", "2023-01-01 11:20:00"]
            ),
        }
    )
    df_engineered = engineer_features(df)
    assert "PU_DO" in df_engineered.columns
    assert "duration" in df_engineered.columns


def test_split_data():
    df = pd.DataFrame(
        {
            "PU_DO": ["10_15", "20_25", "30_35", "40_45"],
            "trip_distance": [1.0, 2.0, 3.0, 4.0],
            "duration": [10.0, 12.0, 15.0, 20.0],
        }
    )
    df_train, df_val = split_data(df)
    assert len(df_train) > 0
    assert len(df_val) > 0


@patch("prodml.train.load_features")
def test_train_main(mock_load_features, tmp_path, monkeypatch):
    # Build small fake feature matrices/targets that skip real parquet I/O
    # and match what load_features() would normally hand back to main().
    rows = [
        {"PU_DO": "10_15", "trip_distance": 1.2},
        {"PU_DO": "20_25", "trip_distance": 3.4},
        {"PU_DO": "30_35", "trip_distance": 2.5},
        {"PU_DO": "40_45", "trip_distance": 5.0},
        {"PU_DO": "50_55", "trip_distance": 1.1},
    ]
    dv = DictVectorizer()
    X_train = dv.fit_transform(rows[:4])
    X_val = dv.transform(rows[4:])
    y_train = np.array([10.0, 12.0, 15.0, 20.0])
    y_val = np.array([11.0])

    mock_load_features.return_value = (X_train, X_val, y_train, y_val, dv)

    # Route MLflow to a throwaway local SQLite store so this test never
    # touches a real tracking server or a persistent ./mlruns directory.
    # (A plain filesystem tracking URI is deprecated as of MLflow 3.13 and
    # raises unless MLFLOW_ALLOW_FILE_STORE=true is set, so SQLite is used
    # here instead of file://.)
    monkeypatch.setattr(
        "prodml.config.settings.MLFLOW_TRACKING_URL", f"sqlite:///{tmp_path}/mlflow.db"
    )

    output_dir = tmp_path / "models"

    # train.main() reads input_dir/output_dir from argv (parsed with
    # argparse.parse_known_args), so pass them explicitly instead of
    # relying on sys.argv, which pytest's own flags would otherwise pollute.
    train_main(argv=["unused-input-dir", str(output_dir)])

    assert (output_dir / "model.pkl").exists()

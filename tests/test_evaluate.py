from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from prodml.evaluate import evaluate_model, main


def test_evaluate_model():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([10.0, 20.0])

    mock_dv = MagicMock()
    mock_dv.transform.return_value = np.array([[1.0], [2.0]])

    test_df = pd.DataFrame(
        {
            "PU_DO": ["10_15", "20_25"],
            "trip_distance": [1.5, 2.5],
            "duration": [12.0, 18.0],
        }
    )

    metrics = evaluate_model(mock_model, mock_dv, test_df)
    assert "test_mae" in metrics
    assert "test_rmse" in metrics
    assert isinstance(metrics["test_mae"], float)


@patch("prodml.evaluate.pickle.load")
@patch("prodml.evaluate.pd.read_parquet")
@patch("prodml.evaluate.json.dump")
@patch("builtins.open", new_callable=MagicMock)
def test_main_evaluate(
    mock_open, mock_json_dump, mock_read_parquet, mock_pickle_load, tmp_path
):
    mock_dv = MagicMock()
    # جعل الـ transform يعيد مصفوفة ثنائية الأبعاد حقيقية لتجنب الـ TypeError
    mock_dv.transform.return_value = np.array([[1.0]])

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([12.0])

    mock_pickle_load.return_value = {"model": mock_model, "dv": mock_dv}
    mock_read_parquet.return_value = pd.DataFrame(
        {"PU_DO": ["10_15"], "trip_distance": [1.5], "duration": [12.0]}
    )

    with patch("sys.argv", ["evaluate.py", "model.pkl", "data_dir", str(tmp_path)]):
        main()

    mock_json_dump.assert_called_once()

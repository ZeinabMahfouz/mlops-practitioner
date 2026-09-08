from unittest.mock import MagicMock, patch
from prodml.registry import promote_best_model


@patch("prodml.registry.MlflowClient")
@patch("prodml.registry.mlflow.register_model")
@patch("prodml.registry.mlflow.set_tracking_uri")
def test_promote_best_model(mock_set_uri, mock_register_model, mock_client_cls):
    # Mock client instance and methods
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    # Mock experiment
    mock_exp = MagicMock()
    mock_exp.experiment_id = "123"
    mock_client.get_experiment_by_name.return_value = mock_exp

    # Mock runs
    mock_run = MagicMock()
    mock_run.info.run_id = "run_abc"
    mock_run.data.metrics.get.return_value = 2.3456
    mock_client.search_runs.return_value = [mock_run]

    # Mock registered model version
    mock_version = MagicMock()
    mock_version.version = "1"
    mock_register_model.return_value = mock_version

    # Execute
    result = promote_best_model()

    # Assertions
    mock_set_uri.assert_called_once()
    mock_client.get_experiment_by_name.assert_called_once_with(
        "ride-duration-prediction"
    )
    mock_client.search_runs.assert_called_once()
    mock_register_model.assert_called_once()
    mock_client.transition_model_version_stage.assert_called_once()
    assert result.version == "1"


@patch("prodml.registry.MlflowClient")
@patch("prodml.registry.mlflow.set_tracking_uri")
def test_promote_best_model_no_experiment(mock_set_uri, mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_client.get_experiment_by_name.return_value = None

    result = promote_best_model()
    assert result is None

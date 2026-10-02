import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pythonbasictools.experiment_utils.output_folder import ExperimentState

from fermiml_framework_classification.datasets.bars_and_stripes import BarsAndStripes
from fermiml_framework_classification.experiments.configs.dataset_config import (
    DatasetConfig,
)
from fermiml_framework_classification.experiments.configs.model_config import (
    ModelConfig,
)
from fermiml_framework_classification.experiments.configs.single_experiment_config import (
    SingleExperimentConfig,
)
from fermiml_framework_classification.experiments.single_experiment import (
    SingleExperiment,
)
from fermiml_framework_classification.models.iqp_kernel_classifier import (
    IQPKernelClassifier,
)


class TestSingleExperiment:
    @pytest.fixture
    def config(self):
        return SingleExperimentConfig(
            dataset=DatasetConfig(dataset_name="bars-and-stripes"),
            model=ModelConfig(name="IQP", hyperparameters={}),
        )

    @pytest.fixture
    def mock_output_folder(self, monkeypatch, tmp_path):
        mock = MagicMock()
        mock.path = tmp_path
        mock.metadata_file.raveled_state = {}
        mock.data_file.raveled_state = {}
        monkeypatch.setattr(
            "fermiml_framework_classification.experiments.single_experiment.OutputFolder",
            MagicMock(return_value=mock),
        )
        return mock

    @pytest.fixture
    def experiment(self, config, mock_output_folder):
        return SingleExperiment(config, root_output_folder="/tmp/test", output_folder="run_0")

    @pytest.fixture
    def mock_dataset(self):
        ds = MagicMock()
        ds.x_train = np.ones((4, 2))
        ds.y_train = np.array([0, 1, 0, 1])
        ds.x_val = np.ones((2, 2))
        ds.y_val = np.array([0, 1])
        ds.x_test = np.ones((2, 2))
        ds.y_test = np.array([1, 0])
        return ds

    @pytest.fixture
    def mock_model(self):
        m = MagicMock()
        m.predict.side_effect = lambda x: np.zeros(len(x), dtype=int)
        return m

    # --- __init__ ---

    def test_config_stored(self, experiment, config):
        assert experiment.config is config

    def test_model_cls_resolved(self, experiment):
        assert experiment._model_cls is IQPKernelClassifier

    def test_dataset_cls_resolved(self, experiment):
        assert experiment._dataset_cls is BarsAndStripes

    def test_custom_root_output_folder(self, experiment):
        assert experiment._root_output_folder == Path("/tmp/test")

    def test_default_root_output_folder(self, config, mock_output_folder):
        exp = SingleExperiment(config)
        assert exp._root_output_folder == Path(os.getcwd()) / "data"

    # --- properties ---

    def test_config_property(self, experiment, config):
        assert experiment.config is config

    def test_output_folder_property(self, experiment, mock_output_folder):
        assert experiment.output_folder is mock_output_folder

    # --- _get_model_cls / _get_dataset_cls ---

    def test_get_model_cls_returns_iqp(self, experiment):
        assert experiment._get_model_cls() is IQPKernelClassifier

    def test_get_model_cls_non_first_in_list(self, experiment):
        from fermiml_framework_classification.models.fermi_vanilla_kernel_classifier import (
            FermiVanillaKernelClassifier,
        )

        experiment._config.model.name = "Vanilla"
        assert experiment._get_model_cls() is FermiVanillaKernelClassifier

    def test_get_dataset_cls_returns_bars_and_stripes(self, experiment):
        assert experiment._get_dataset_cls() is BarsAndStripes

    def test_get_dataset_cls_invalid_raises(self, experiment):
        experiment._config.dataset.dataset_name = "nonexistent"
        with pytest.raises(ValueError, match="not supported"):
            experiment._get_dataset_cls()

    # --- _run_pipeline ---

    def test_run_pipeline_calls_prepare_data(self, experiment, mock_dataset, mock_model):
        experiment._dataset_cls = MagicMock(return_value=mock_dataset)
        experiment._model_cls = MagicMock(return_value=mock_model)
        experiment._run_pipeline()
        mock_dataset.prepare_data.assert_called_once()

    def test_run_pipeline_fits_model(self, experiment, mock_dataset, mock_model):
        experiment._dataset_cls = MagicMock(return_value=mock_dataset)
        experiment._model_cls = MagicMock(return_value=mock_model)
        experiment._run_pipeline()
        mock_model.fit.assert_called_once_with(mock_dataset.x_train, mock_dataset.y_train)

    def test_run_pipeline_predicts_all_splits(self, experiment, mock_dataset, mock_model):
        experiment._dataset_cls = MagicMock(return_value=mock_dataset)
        experiment._model_cls = MagicMock(return_value=mock_model)
        experiment._run_pipeline()
        assert mock_model.predict.call_count == 3

    def test_run_pipeline_stores_metrics(self, experiment, mock_dataset, mock_model, mock_output_folder):
        experiment._dataset_cls = MagicMock(return_value=mock_dataset)
        experiment._model_cls = MagicMock(return_value=mock_model)
        experiment._run_pipeline()
        kwargs = mock_output_folder.update_data.call_args.kwargs
        expected = {
            "train_accuracy",
            "train_f1",
            "train_recall",
            "train_precision",
            "val_accuracy",
            "val_f1",
            "val_recall",
            "val_precision",
            "test_accuracy",
            "test_f1",
            "test_recall",
            "test_precision",
        }
        assert expected.issubset(kwargs.keys())

    def test_run_pipeline_passes_dataset_config_options(self, experiment, mock_dataset, mock_model):
        dataset_cls_mock = MagicMock(return_value=mock_dataset)
        experiment._dataset_cls = dataset_cls_mock
        experiment._model_cls = MagicMock(return_value=mock_model)
        experiment._run_pipeline()
        _, kwargs = dataset_cls_mock.call_args
        assert kwargs["cv_split_id"] == experiment.config.dataset.split_id
        assert kwargs["cv_splits"] == experiment.config.dataset.cv_splits

    # --- run ---

    def test_run_returns_zero(self, experiment, monkeypatch):
        monkeypatch.setattr(experiment, "_run_pipeline", MagicMock())
        assert experiment.run() == 0

    def test_run_sets_state_finished_on_success(self, experiment, mock_output_folder, monkeypatch):
        monkeypatch.setattr(experiment, "_run_pipeline", MagicMock())
        experiment.run()
        assert mock_output_folder.state == ExperimentState.FINISHED

    def test_run_sets_state_failed_on_error(self, experiment, mock_output_folder, monkeypatch):
        monkeypatch.setattr(experiment, "_run_pipeline", MagicMock(side_effect=RuntimeError("boom")))
        experiment.raise_errors = False
        experiment.run()
        assert mock_output_folder.state == ExperimentState.FAILED

    def test_run_does_not_raise_when_flag_unset(self, experiment, monkeypatch):
        monkeypatch.setattr(experiment, "_run_pipeline", MagicMock(side_effect=RuntimeError("boom")))
        experiment.raise_errors = False
        assert experiment.run() == 0

    def test_run_raises_when_flag_set(self, experiment, monkeypatch):
        monkeypatch.setattr(experiment, "_run_pipeline", MagicMock(side_effect=RuntimeError("boom")))
        experiment.raise_errors = True
        with pytest.raises(RuntimeError, match="boom"):
            experiment.run()

    def test_run_logs_metadata_and_output(self, experiment, monkeypatch):
        log_calls = []
        monkeypatch.setattr(experiment, "_run_pipeline", MagicMock())
        experiment.log_func = log_calls.append
        experiment.run()
        full_log = "\n".join(log_calls)
        assert "METADATA" in full_log
        assert "Output" in full_log

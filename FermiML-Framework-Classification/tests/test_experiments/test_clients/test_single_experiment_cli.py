import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from fermiml_framework_classification.experiments.clients.single_experiment_cli import (
    single_experiment_cli,
)
from fermiml_framework_classification.experiments.configs.dataset_config import (
    DatasetConfig,
)
from fermiml_framework_classification.experiments.configs.model_config import (
    ModelConfig,
)
from fermiml_framework_classification.experiments.configs.single_experiment_config import (
    SingleExperimentConfig,
)


class TestSingleExperimentCLI:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.runner = CliRunner()
        yield

    @pytest.fixture
    def config_instance(self):
        return SingleExperimentConfig(
            dataset=DatasetConfig(dataset_name="bars-and-stripes"),
            model=ModelConfig(name="IQP", hyperparameters={}),
        )

    @pytest.fixture(autouse=True)
    def config_yaml(self, config_instance, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_instance.to_yaml_file(config_path)
        yield config_path

    @pytest.fixture(autouse=True)
    def mock_experiment(self, monkeypatch):
        instance_mock = MagicMock()
        instance_mock.run.return_value = 0
        cls_mock = MagicMock(return_value=instance_mock)
        monkeypatch.setattr(
            "fermiml_framework_classification.experiments.clients.single_experiment_cli.SingleExperiment",
            cls_mock,
        )
        return cls_mock

    def test_cli_runs(self, mock_experiment, config_yaml):
        result = self.runner.invoke(single_experiment_cli, [str(config_yaml)])
        assert result.exit_code == 0, f"CLI did not exit cleanly: {result.output}"
        mock_experiment.assert_called_once()
        mock_experiment.return_value.run.assert_called_once()

    def test_cli_output_contains_finished(self, config_yaml):
        result = self.runner.invoke(single_experiment_cli, [str(config_yaml)])
        assert "Finished" in result.output

    def test_cli_missing_config_file(self):
        result = self.runner.invoke(single_experiment_cli, [])
        assert result.exit_code != 0

    def test_cli_nonexistent_config_file(self):
        result = self.runner.invoke(single_experiment_cli, ["nonexistent.yaml"])
        assert result.exit_code != 0

    def test_cli_passes_root_output_folder(self, mock_experiment, config_yaml, tmp_path):
        root = str(tmp_path / "output")
        result = self.runner.invoke(single_experiment_cli, [str(config_yaml), "--root_output_folder", root])
        assert result.exit_code == 0, result.output
        _, kwargs = mock_experiment.call_args
        assert str(kwargs["root_output_folder"]) == root

    def test_cli_passes_output_folder(self, mock_experiment, config_yaml, tmp_path):
        out = str(tmp_path / "run_42")
        result = self.runner.invoke(single_experiment_cli, [str(config_yaml), "--output_folder", out])
        assert result.exit_code == 0, result.output
        _, kwargs = mock_experiment.call_args
        assert str(kwargs["output_folder"]) == out

    def test_cli_raise_errors_default_false(self, mock_experiment, config_yaml):
        self.runner.invoke(single_experiment_cli, [str(config_yaml)])
        _, kwargs = mock_experiment.call_args
        assert kwargs["raise_errors"] is False

    def test_cli_raise_errors_flag(self, mock_experiment, config_yaml):
        result = self.runner.invoke(single_experiment_cli, [str(config_yaml), "--raise_errors"])
        assert result.exit_code == 0, result.output
        _, kwargs = mock_experiment.call_args
        assert kwargs["raise_errors"] is True

    def test_cli_sets_omp_num_threads(self, config_yaml, monkeypatch):
        import os

        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
        self.runner.invoke(single_experiment_cli, [str(config_yaml)])
        assert os.environ.get("OMP_NUM_THREADS") == "4"

    def test_cli_sets_mkl_num_threads(self, config_yaml, monkeypatch):
        import os

        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        self.runner.invoke(single_experiment_cli, [str(config_yaml)])
        assert os.environ.get("MKL_NUM_THREADS") == "8"

    def test_cli_loads_correct_config(self, mock_experiment, config_instance, config_yaml):
        self.runner.invoke(single_experiment_cli, [str(config_yaml)])
        loaded_config = mock_experiment.call_args[0][0]
        assert loaded_config.dataset.dataset_name == config_instance.dataset.dataset_name
        assert loaded_config.model.name == config_instance.model.name

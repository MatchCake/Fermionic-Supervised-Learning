import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from fermihamml.experiments.clients.single_experiment_cli import single_experiment_cli
from fermihamml.experiments.configs.dataset_config import DatasetConfig
from fermihamml.experiments.configs.model_config import ModelConfig
from fermihamml.experiments.configs.opt_config import OptConfig
from fermihamml.experiments.configs.single_experiment_config import (
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
            model=ModelConfig(
                name="NIFIsingCls",
                extra_options={"test_option": 0},
            ),
            dataset_config=DatasetConfig(
                "Cifar2",
                split_id=0,
                seed=0,
                batch_size=32,
            ),
            opt_config=OptConfig(
                automl_iterations=1,
                inner_max_epochs=1,
                max_epochs=1,
                automl_overwrite_fit=True,
            ),
        )

    @pytest.fixture(autouse=True)
    def config_yaml(self, config_instance):
        config_path = Path(".tmp/cli") / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_instance.to_yaml_file(config_path)
        yield config_path
        shutil.rmtree(config_path.parent, ignore_errors=True)

    @pytest.fixture(autouse=True)
    def mock_experiment(self, monkeypatch):
        instance_mock = MagicMock()
        cls_mock = MagicMock(return_value=instance_mock)
        instance_mock.run = MagicMock()
        cls_mock.run = instance_mock.run
        monkeypatch.setattr("fermihamml.experiments.clients.single_experiment_cli.SingleExperiment", cls_mock)
        return cls_mock

    # @patch("fermihamml.experiments.clients.single_experiment_cli.SingleExperiment")
    def test_single_experiment_cli_runs(self, mock_experiment, config_yaml):
        result = self.runner.invoke(single_experiment_cli, [str(config_yaml)])
        assert result.exit_code == 0, f"CLI did not exit cleanly: {result.output}"
        mock_experiment.assert_called_once()
        mock_experiment.run.assert_called_once()

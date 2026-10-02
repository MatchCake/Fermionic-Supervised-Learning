from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from pythonbasictools.experiment_utils.output_folder import ExperimentState

from fermiml_framework_classification.experiments.clients.compute_ith_cli import (
    compute_ith_cli,
)
from fermiml_framework_classification.experiments.configs.datasets_config import (
    DatasetsConfig,
    DatasetsConfigItem,
)
from fermiml_framework_classification.experiments.configs.experiment_config import (
    ExperimentConfig,
)
from fermiml_framework_classification.experiments.configs.model_hps_config import (
    ModelHpsConfig,
)


def make_exp_config_yaml(tmp_path):
    config = ExperimentConfig(
        experiment_name="test_exp",
        datasets=DatasetsConfig(
            datasets=[DatasetsConfigItem(name="bars-and-stripes")],
            cv_splits=1,
        ),
        models=[ModelHpsConfig(name="IQP", hyperparameters={"svm_c": [1]})],
    )
    path = tmp_path / "config.yaml"
    config.to_yaml_file(path)
    return path


class TestComputeIthCLI:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.runner = CliRunner()

    @pytest.fixture
    def config_yaml(self, tmp_path):
        return make_exp_config_yaml(tmp_path)

    @pytest.fixture(autouse=True)
    def mock_output_folder(self, monkeypatch):
        mock_instance = MagicMock()
        mock_instance.state = ExperimentState.RUNNING
        monkeypatch.setattr(
            "fermiml_framework_classification.experiments.clients.compute_ith_cli.OutputFolder",
            MagicMock(return_value=mock_instance),
        )
        return mock_instance

    @pytest.fixture(autouse=True)
    def mock_experiment(self, monkeypatch):
        instance = MagicMock()
        instance.run.return_value = 0
        cls = MagicMock(return_value=instance)
        monkeypatch.setattr(
            "fermiml_framework_classification.experiments.clients.compute_ith_cli.SingleExperiment",
            cls,
        )
        return cls

    def test_cli_runs(self, config_yaml, mock_experiment, tmp_path):
        result = self.runner.invoke(
            compute_ith_cli, ["0", "--exp_config_file", str(config_yaml), "--root_dir", str(tmp_path)]
        )
        assert result.exit_code == 0, result.output

    def test_cli_calls_experiment_run(self, config_yaml, mock_experiment, tmp_path):
        self.runner.invoke(compute_ith_cli, ["0", "--exp_config_file", str(config_yaml), "--root_dir", str(tmp_path)])
        mock_experiment.return_value.run.assert_called_once()

    def test_cli_output_contains_finished(self, config_yaml, mock_experiment, tmp_path):
        result = self.runner.invoke(
            compute_ith_cli, ["0", "--exp_config_file", str(config_yaml), "--root_dir", str(tmp_path)]
        )
        assert "Finished" in result.output

    def test_cli_out_of_range_config_id(self, config_yaml, mock_experiment, tmp_path):
        result = self.runner.invoke(
            compute_ith_cli, ["99", "--exp_config_file", str(config_yaml), "--root_dir", str(tmp_path)]
        )
        assert result.exit_code != 0

    def test_cli_nonexistent_config_file(self, tmp_path):
        result = self.runner.invoke(compute_ith_cli, ["0", "--exp_config_file", str(tmp_path / "missing.yaml")])
        assert result.exit_code != 0

    def test_cli_sets_omp_num_threads(self, config_yaml, mock_experiment, tmp_path, monkeypatch):
        import os

        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
        self.runner.invoke(compute_ith_cli, ["0", "--exp_config_file", str(config_yaml), "--root_dir", str(tmp_path)])
        assert os.environ.get("OMP_NUM_THREADS") == "4"

    def test_cli_no_args_raises_usage_error(self, config_yaml, tmp_path):
        result = self.runner.invoke(
            compute_ith_cli, ["--exp_config_file", str(config_yaml), "--root_dir", str(tmp_path)]
        )
        assert result.exit_code != 0
        assert "Provide either CONFIG_ID or --config_hash" in result.output

    def test_cli_runs_by_hash(self, config_yaml, mock_experiment, tmp_path):
        from fermiml_framework_classification.experiments.configs.experiment_config import (
            ExperimentConfig,
        )

        configs = list(ExperimentConfig.from_yaml_file(config_yaml).to_single_experiment_configs())
        valid_hash = configs[0].hash
        result = self.runner.invoke(
            compute_ith_cli,
            ["--config_hash", valid_hash, "--exp_config_file", str(config_yaml), "--root_dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output

    def test_cli_invalid_hash_raises_error(self, config_yaml, mock_experiment, tmp_path):
        result = self.runner.invoke(
            compute_ith_cli,
            ["--config_hash", "deadbeef", "--exp_config_file", str(config_yaml), "--root_dir", str(tmp_path)],
        )
        assert result.exit_code != 0

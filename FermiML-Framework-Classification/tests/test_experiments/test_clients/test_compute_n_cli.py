from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from pythonbasictools.experiment_utils.output_folder import ExperimentState

from fermiml_framework_classification.experiments.clients.compute_n_cli import (
    compute_n_cli,
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
        models=[ModelHpsConfig(name="IQP", hyperparameters={"svm_c": [1, 10]})],
    )
    path = tmp_path / "config.yaml"
    config.to_yaml_file(path)
    return path


class TestComputeNCLI:
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
            "fermiml_framework_classification.experiments.clients.compute_n_cli.OutputFolder",
            MagicMock(return_value=mock_instance),
        )
        return mock_instance

    def test_cli_runs(self, config_yaml, tmp_path):
        out_file = tmp_path / "n.txt"
        result = self.runner.invoke(
            compute_n_cli,
            [
                "--exp_config_file",
                str(config_yaml),
                "--root_dir",
                str(tmp_path),
                "--out_file",
                str(out_file),
            ],
        )
        assert result.exit_code == 0, result.output

    def test_cli_writes_count_to_file(self, config_yaml, tmp_path):
        out_file = tmp_path / "n.txt"
        self.runner.invoke(
            compute_n_cli,
            [
                "--exp_config_file",
                str(config_yaml),
                "--root_dir",
                str(tmp_path),
                "--out_file",
                str(out_file),
            ],
        )
        assert out_file.read_text() == "2"

    def test_cli_outputs_count(self, config_yaml, tmp_path):
        out_file = tmp_path / "n.txt"
        result = self.runner.invoke(
            compute_n_cli,
            [
                "--exp_config_file",
                str(config_yaml),
                "--root_dir",
                str(tmp_path),
                "--out_file",
                str(out_file),
            ],
        )
        assert "2" in result.output

    def test_cli_nonexistent_config_file(self, tmp_path):
        result = self.runner.invoke(compute_n_cli, ["--exp_config_file", str(tmp_path / "missing.yaml")])
        assert result.exit_code != 0

    def test_cli_finished_jobs_excluded(self, config_yaml, tmp_path, monkeypatch):
        mock_instance = MagicMock()
        mock_instance.state = ExperimentState.FINISHED
        monkeypatch.setattr(
            "fermiml_framework_classification.experiments.clients.compute_n_cli.OutputFolder",
            MagicMock(return_value=mock_instance),
        )
        out_file = tmp_path / "n.txt"
        self.runner.invoke(
            compute_n_cli,
            [
                "--exp_config_file",
                str(config_yaml),
                "--root_dir",
                str(tmp_path),
                "--out_file",
                str(out_file),
            ],
        )
        assert out_file.read_text() == "0"

    def test_cli_sets_omp_num_threads(self, config_yaml, tmp_path, monkeypatch):
        import os

        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
        out_file = tmp_path / "n.txt"
        self.runner.invoke(
            compute_n_cli,
            [
                "--exp_config_file",
                str(config_yaml),
                "--root_dir",
                str(tmp_path),
                "--out_file",
                str(out_file),
            ],
        )
        assert os.environ.get("OMP_NUM_THREADS") == "2"

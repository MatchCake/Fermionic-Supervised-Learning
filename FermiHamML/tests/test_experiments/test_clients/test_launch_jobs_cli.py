import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from fermihamml.experiments.clients.launch_jobs_cli import launch_jobs_cli
from fermihamml.experiments.configs.dataset_config import DatasetConfig
from fermihamml.experiments.configs.datasets_config import DatasetsConfig
from fermihamml.experiments.configs.experiment_config import ExperimentConfig
from fermihamml.experiments.configs.model_config import ModelConfig
from fermihamml.experiments.configs.opt_config import OptConfig


class TestLaunchJobsCli:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.runner = CliRunner()
        yield

    @pytest.fixture
    def config_instance(self):
        return ExperimentConfig(
            experiment_name="test_experiment",
            datasets=DatasetsConfig(
                names=["test_dataset"],
                num_split_id=1,
                batch_sizes=[32],
                seed=0,
            ),
            models=[
                ModelConfig(
                    name="NIFIsingCls",
                    extra_options={"test_option": 0},
                )
            ],
            opt_config=OptConfig(
                automl_iterations=1,
                inner_max_epochs=1,
                max_epochs=1,
                automl_overwrite_fit=True,
            ),
        )

    @pytest.fixture(autouse=True)
    def config_yaml(self, config_instance):
        config_path = Path(".tmp/launch_jobs_cli") / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_instance.to_yaml_file(config_path)
        yield config_path
        shutil.rmtree(config_path.parent, ignore_errors=True)

    @pytest.fixture(autouse=True)
    def mock_experiment(self, monkeypatch):
        os_mock = MagicMock()
        monkeypatch.setattr("fermihamml.experiments.clients.launch_jobs_cli.os", os_mock)
        return os_mock

    def test_cli_runs_with_uv_and_system(self, mock_experiment, config_yaml):
        result = self.runner.invoke(
            launch_jobs_cli,
            [
                str(config_yaml),
                "--root_dir",
                ".tmp/launch_jobs_cli",
                "--cmd",
                "uv",
                "--out_file",
                None,
            ],
        )
        assert result.exit_code == 0, f"CLI did not exit cleanly: {result.output}"
        mock_experiment.system.assert_called(), "Expected os.system to be called to submit jobs"
        submitted_cmds = [call.args[0] for call in mock_experiment.system.call_args_list]
        assert any("uv" in cmd for cmd in submitted_cmds), f"Expected 'uv' in submitted commands, got: {submitted_cmds}"
        assert not Path(".tmp/launch_jobs_cli/cmds.txt").exists(), "Expected cmds.txt to be created"

    def test_cli_runs_with_uv_and_out_file(self, mock_experiment, config_yaml):
        result = self.runner.invoke(
            launch_jobs_cli,
            [
                str(config_yaml),
                "--root_dir",
                ".tmp/launch_jobs_cli",
                "--cmd",
                "uv",
                "--out_file",
                ".tmp/launch_jobs_cli/cmds.txt",
            ],
        )
        assert result.exit_code == 0, f"CLI did not exit cleanly: {result.output}"
        mock_experiment.system.assert_not_called(), "Expected os.system to NOT be called when out_file is provided"
        assert Path(".tmp/launch_jobs_cli/cmds.txt").exists(), "Expected cmds.txt to be created"
        with open(".tmp/launch_jobs_cli/cmds.txt", "r") as f:
            cmds_content = f.read()
        assert "uv run" in cmds_content, f"Expected 'uv run' in cmds.txt content, got: {cmds_content}"

    def test_cli_runs_with_sbatch_and_system(self, mock_experiment, config_yaml):
        result = self.runner.invoke(
            launch_jobs_cli,
            [
                str(config_yaml),
                "--root_dir",
                ".tmp/launch_jobs_cli",
                "--cmd",
                "sbatch",
                "--out_file",
                None,
            ],
        )
        assert result.exit_code == 0, f"CLI did not exit cleanly: {result.output}"
        mock_experiment.system.assert_called(), "Expected os.system to be called to submit jobs"
        submitted_cmds = [call.args[0] for call in mock_experiment.system.call_args_list]
        assert any(
            "sbatch" in cmd for cmd in submitted_cmds
        ), f"Expected 'sbatch' in submitted commands, got: {submitted_cmds}"
        assert not Path(".tmp/launch_jobs_cli/cmds.txt").exists(), "Expected cmds.txt to be created"

    def test_cli_runs_with_sbatch_and_out_file(self, mock_experiment, config_yaml):
        result = self.runner.invoke(
            launch_jobs_cli,
            [
                str(config_yaml),
                "--root_dir",
                ".tmp/launch_jobs_cli",
                "--cmd",
                "sbatch",
                "--out_file",
                ".tmp/launch_jobs_cli/cmds.txt",
            ],
        )
        assert result.exit_code == 0, f"CLI did not exit cleanly: {result.output}"
        mock_experiment.system.assert_not_called(), "Expected os.system to NOT be called when out_file is provided"
        assert Path(".tmp/launch_jobs_cli/cmds.txt").exists(), "Expected cmds.txt to be created"
        with open(".tmp/launch_jobs_cli/cmds.txt", "r") as f:
            cmds_content = f.read()
        assert "sbatch" in cmds_content, f"Expected 'sbatch' in cmds.txt content, got: {cmds_content}"

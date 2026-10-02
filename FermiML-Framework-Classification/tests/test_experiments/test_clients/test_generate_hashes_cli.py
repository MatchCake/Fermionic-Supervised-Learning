import json

import pytest
from click.testing import CliRunner

from fermiml_framework_classification.experiments.clients.generate_hashes_cli import (
    generate_hashes_cli,
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


def make_exp_config_yaml(tmp_path, *, cv_splits=2, svm_c=None, experiment_name="test_exp"):
    config = ExperimentConfig(
        experiment_name=experiment_name,
        datasets=DatasetsConfig(
            datasets=[DatasetsConfigItem(name="bars-and-stripes")],
            cv_splits=cv_splits,
        ),
        models=[ModelHpsConfig(name="IQP", hyperparameters={"svm_c": svm_c or [1]})],
    )
    path = tmp_path / "config.yaml"
    config.to_yaml_file(path)
    return path, config


class TestGenerateHashesCLI:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.runner = CliRunner()

    @pytest.fixture
    def config_yaml(self, tmp_path):
        path, _ = make_exp_config_yaml(tmp_path)
        return path

    def test_cli_exits_zero(self, config_yaml, tmp_path):
        output = tmp_path / "hashes.json"
        result = self.runner.invoke(
            generate_hashes_cli,
            ["--config_file", str(config_yaml), "--output", str(output)],
        )
        assert result.exit_code == 0, result.output

    def test_output_file_created(self, config_yaml, tmp_path):
        output = tmp_path / "hashes.json"
        self.runner.invoke(
            generate_hashes_cli,
            ["--config_file", str(config_yaml), "--output", str(output)],
        )
        assert output.exists()

    def test_output_is_valid_json(self, config_yaml, tmp_path):
        output = tmp_path / "hashes.json"
        self.runner.invoke(
            generate_hashes_cli,
            ["--config_file", str(config_yaml), "--output", str(output)],
        )
        data = json.loads(output.read_text())
        assert "experiment_name" in data
        assert "configs" in data

    def test_experiment_name_matches_config(self, tmp_path):
        config_path, _ = make_exp_config_yaml(tmp_path, experiment_name="my_experiment")
        output = tmp_path / "hashes.json"
        self.runner.invoke(
            generate_hashes_cli,
            ["--config_file", str(config_path), "--output", str(output)],
        )
        data = json.loads(output.read_text())
        assert data["experiment_name"] == "my_experiment"

    def test_config_count_matches_hyperparameter_grid(self, tmp_path):
        config_path, config = make_exp_config_yaml(tmp_path, svm_c=[0.1, 1, 10])
        output = tmp_path / "hashes.json"
        self.runner.invoke(
            generate_hashes_cli,
            ["--config_file", str(config_path), "--output", str(output)],
        )
        data = json.loads(output.read_text())
        expected = len(list(config.to_single_experiment_configs()))
        assert len(data["configs"]) == expected

    def test_each_entry_has_required_keys(self, config_yaml, tmp_path):
        output = tmp_path / "hashes.json"
        self.runner.invoke(
            generate_hashes_cli,
            ["--config_file", str(config_yaml), "--output", str(output)],
        )
        data = json.loads(output.read_text())
        for entry in data["configs"]:
            assert "hash" in entry
            assert "mem_mb" in entry
            assert "runtime_min" in entry

    def test_hashes_are_unique(self, tmp_path):
        config_path, _ = make_exp_config_yaml(tmp_path, svm_c=[0.1, 1, 10])
        output = tmp_path / "hashes.json"
        self.runner.invoke(
            generate_hashes_cli,
            ["--config_file", str(config_path), "--output", str(output)],
        )
        data = json.loads(output.read_text())
        hashes = [e["hash"] for e in data["configs"]]
        assert len(hashes) == len(set(hashes))

    def test_hashes_match_experiment_config(self, tmp_path):
        config_path, config = make_exp_config_yaml(tmp_path, svm_c=[1, 10])
        output = tmp_path / "hashes.json"
        self.runner.invoke(
            generate_hashes_cli,
            ["--config_file", str(config_path), "--output", str(output)],
        )
        data = json.loads(output.read_text())
        expected_hashes = {c.hash for c in config.to_single_experiment_configs()}
        actual_hashes = {e["hash"] for e in data["configs"]}
        assert actual_hashes == expected_hashes

    def test_output_directory_created_if_missing(self, config_yaml, tmp_path):
        output = tmp_path / "nested" / "dir" / "hashes.json"
        result = self.runner.invoke(
            generate_hashes_cli,
            ["--config_file", str(config_yaml), "--output", str(output)],
        )
        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_nonexistent_config_file_fails(self, tmp_path):
        output = tmp_path / "hashes.json"
        result = self.runner.invoke(
            generate_hashes_cli,
            ["--config_file", str(tmp_path / "missing.yaml"), "--output", str(output)],
        )
        assert result.exit_code != 0

    def test_output_contains_experiment_name_in_stdout(self, config_yaml, tmp_path):
        output = tmp_path / "hashes.json"
        result = self.runner.invoke(
            generate_hashes_cli,
            ["--config_file", str(config_yaml), "--output", str(output)],
        )
        assert "test_exp" in result.output

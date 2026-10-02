import json

from click.testing import CliRunner

from fermiml_framework_regression.experiments.clients.generate_hashes_cli import generate_hashes_cli
from fermiml_framework_regression.experiments.configs.experiment_config import ExperimentConfig
from tests.test_experiments.test_clients.conftest import EXPECTED_JOBS


class TestGenerateHashesCli:
    """``workflow/hashes.json`` is the job list the Snakefile parses at load time.

    Every target the workflow submits comes from this file, so its shape is a contract between
    the container (which writes it) and the login node (which reads it, without being able to
    import the package). A missing key here is a workflow that cannot parse its own DAG.
    """

    def test_writes_one_entry_per_job(self, config_file, tmp_path):
        output = tmp_path / "hashes.json"
        result = CliRunner().invoke(generate_hashes_cli, ["--config_file", str(config_file), "--output", str(output)])
        assert result.exit_code == 0, result.output
        payload = json.loads(output.read_text())
        assert payload["experiment_name"] == "cli_test_experiment"
        assert len(payload["configs"]) == EXPECTED_JOBS

    def test_every_entry_carries_what_the_snakefile_reads(self, config_file, tmp_path):
        output = tmp_path / "hashes.json"
        CliRunner().invoke(generate_hashes_cli, ["--config_file", str(config_file), "--output", str(output)])
        for entry in json.loads(output.read_text())["configs"]:
            # hash keys the rule's output path; mem_mb and runtime_min become its resources;
            # model drives submission priority.
            assert set(entry) == {"hash", "model", "dataset", "mem_mb", "runtime_min"}
            assert len(entry["hash"]) == 64
            assert entry["mem_mb"] > 0
            assert entry["runtime_min"] > 0

    def test_resources_follow_their_model(self, config_file, tmp_path):
        output = tmp_path / "hashes.json"
        CliRunner().invoke(generate_hashes_cli, ["--config_file", str(config_file), "--output", str(output)])
        by_model = {entry["model"]: entry for entry in json.loads(output.read_text())["configs"]}
        assert by_model["AngleNIFGP"]["mem_mb"] == 8192
        assert by_model["ClassicalGP"]["mem_mb"] == 4096

    def test_hashes_are_unique_and_sorted(self, config_file, tmp_path):
        output = tmp_path / "hashes.json"
        CliRunner().invoke(generate_hashes_cli, ["--config_file", str(config_file), "--output", str(output)])
        hashes = [entry["hash"] for entry in json.loads(output.read_text())["configs"]]
        # A collision would make two jobs share a result folder; the sort keeps compute_ith's
        # positional CONFIG_ID stable between invocations.
        assert len(set(hashes)) == len(hashes)
        assert hashes == sorted(hashes)

    def test_hashes_match_the_package_expansion(self, config_file, tmp_path):
        output = tmp_path / "hashes.json"
        CliRunner().invoke(generate_hashes_cli, ["--config_file", str(config_file), "--output", str(output)])
        written = {entry["hash"] for entry in json.loads(output.read_text())["configs"]}
        expected = {c.hash for c in ExperimentConfig.from_yaml_file(config_file).to_single_experiment_configs()}
        assert written == expected

    def test_creates_the_output_directory(self, config_file, tmp_path):
        output = tmp_path / "nested" / "deeper" / "hashes.json"
        result = CliRunner().invoke(generate_hashes_cli, ["--config_file", str(config_file), "--output", str(output)])
        assert result.exit_code == 0, result.output
        assert output.is_file()

    def test_a_missing_config_is_rejected(self, tmp_path):
        result = CliRunner().invoke(
            generate_hashes_cli,
            ["--config_file", str(tmp_path / "absent.yaml"), "--output", str(tmp_path / "h.json")],
        )
        assert result.exit_code != 0

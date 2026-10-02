import json

from click.testing import CliRunner

from fermiml_framework_regression.experiments.clients.compute_ith_cli import compute_ith_cli
from fermiml_framework_regression.experiments.configs.experiment_config import ExperimentConfig

EXPERIMENT = "cli_test_experiment"


def _hashes(config_file):
    """:return: The job hashes in the order compute_ith indexes them. :rtype: list"""
    configs = ExperimentConfig.from_yaml_file(config_file).to_single_experiment_configs()
    return [c.hash for c in sorted(configs, key=lambda c: c.hash)]


class TestComputeIthCli:
    """The entry point every cluster job runs.

    ``workflow/Snakefile`` invokes this once per hash inside the container, so its argument
    handling and its output location are the interface between the workflow and the science.
    """

    def test_runs_the_requested_hash(self, config_file, tmp_path):
        target = _hashes(config_file)[0]
        result = CliRunner().invoke(
            compute_ith_cli,
            ["--config_hash", target, "--exp_config_file", str(config_file), "--root_dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        folder = tmp_path / EXPERIMENT / "results" / target
        assert (folder / "state.finished").is_file()
        payload = json.loads((folder / "run_output.out.json").read_text())["data"]
        assert isinstance(payload["test_r2"], float)

    def test_writes_under_the_experiment_name(self, config_file, tmp_path):
        # The Snakefile's declared output is data/<experiment>/results/<hash>/state.finished;
        # if this layout moves, every rule stops matching its own output.
        target = _hashes(config_file)[0]
        result = CliRunner().invoke(
            compute_ith_cli,
            ["--config_hash", target, "--exp_config_file", str(config_file), "--root_dir", str(tmp_path)],
        )
        # The exit code is asserted because SingleExperiment.__init__ creates the folder before
        # anything runs; without it this passes on a job that failed immediately.
        assert result.exit_code == 0, result.output
        assert (tmp_path / EXPERIMENT / "results" / target / "state.finished").is_file()

    def test_positional_index_selects_the_same_job_as_its_hash(self, config_file, tmp_path):
        # compute_n / an array-job driver address jobs by index; the index must agree with the
        # hash ordering or a rerun computes a different cell than it reports.
        expected = _hashes(config_file)[1]
        result = CliRunner().invoke(
            compute_ith_cli,
            ["1", "--exp_config_file", str(config_file), "--root_dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / EXPERIMENT / "results" / expected / "state.finished").is_file()

    def test_requires_an_index_or_a_hash(self, config_file, tmp_path):
        result = CliRunner().invoke(
            compute_ith_cli, ["--exp_config_file", str(config_file), "--root_dir", str(tmp_path)]
        )
        assert result.exit_code != 0
        assert "Provide either" in result.output

    def test_an_unknown_hash_is_rejected(self, config_file, tmp_path):
        result = CliRunner().invoke(
            compute_ith_cli,
            ["--config_hash", "0" * 64, "--exp_config_file", str(config_file), "--root_dir", str(tmp_path)],
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)

    def test_an_out_of_range_index_is_rejected(self, config_file, tmp_path):
        result = CliRunner().invoke(
            compute_ith_cli, ["999", "--exp_config_file", str(config_file), "--root_dir", str(tmp_path)]
        )
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)

    def test_reads_the_config_from_the_environment(self, config_file, tmp_path):
        # The Snakefile can pass EXP_CONFIG_FILE instead of the flag; both paths must work.
        target = _hashes(config_file)[0]
        result = CliRunner().invoke(
            compute_ith_cli,
            ["--config_hash", target, "--root_dir", str(tmp_path)],
            env={"EXP_CONFIG_FILE": str(config_file)},
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / EXPERIMENT / "results" / target / "state.finished").is_file()

    def test_rerunning_a_finished_hash_is_safe(self, config_file, tmp_path):
        # Snakemake reruns a hash whenever its marker is missing, and --requeue can replay a
        # job that already started; a second run must land on the same answer, not a new one.
        target = _hashes(config_file)[0]
        arguments = [
            "--config_hash", target, "--exp_config_file", str(config_file), "--root_dir", str(tmp_path),
        ]  # fmt: skip
        CliRunner().invoke(compute_ith_cli, arguments)
        folder = tmp_path / EXPERIMENT / "results" / target
        first = json.loads((folder / "run_output.out.json").read_text())["data"]["test_predictions"]
        CliRunner().invoke(compute_ith_cli, arguments)
        second = json.loads((folder / "run_output.out.json").read_text())["data"]["test_predictions"]
        assert first == second

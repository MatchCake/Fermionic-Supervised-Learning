from click.testing import CliRunner

from fermiml_framework_regression.experiments.clients.compute_ith_cli import compute_ith_cli
from fermiml_framework_regression.experiments.clients.compute_n_cli import compute_n_cli
from tests.test_experiments.test_clients.conftest import EXPECTED_JOBS

EXPERIMENT = "cli_test_experiment"


class TestComputeNCli:
    """Counts the jobs still to run.

    Note the side effect this exercises: constructing an ``OutputFolder`` to read a job's state
    creates that folder and a ``state.unknown`` in it. Counting the sweep therefore materialises
    the results tree, which is worth knowing before pointing it at a real one.
    """

    def test_counts_every_job_when_nothing_has_run(self, config_file, tmp_path):
        out_file = tmp_path / "n.txt"
        result = CliRunner().invoke(
            compute_n_cli,
            ["--exp_config_file", str(config_file), "--root_dir", str(tmp_path), "--out_file", str(out_file)],
        )
        assert result.exit_code == 0, result.output
        assert out_file.read_text().strip() == str(EXPECTED_JOBS)

    def test_finished_jobs_are_not_counted(self, config_file, tmp_path):
        from fermiml_framework_regression.experiments.configs.experiment_config import ExperimentConfig

        target = sorted(
            ExperimentConfig.from_yaml_file(config_file).to_single_experiment_configs(), key=lambda c: c.hash
        )[0].hash
        CliRunner().invoke(
            compute_ith_cli,
            ["--config_hash", target, "--exp_config_file", str(config_file), "--root_dir", str(tmp_path)],
        )
        out_file = tmp_path / "n.txt"
        CliRunner().invoke(
            compute_n_cli,
            ["--exp_config_file", str(config_file), "--root_dir", str(tmp_path), "--out_file", str(out_file)],
        )
        assert out_file.read_text().strip() == str(EXPECTED_JOBS - 1)

    def test_reports_the_count_on_stdout(self, config_file, tmp_path):
        result = CliRunner().invoke(
            compute_n_cli,
            [
                "--exp_config_file",
                str(config_file),
                "--root_dir",
                str(tmp_path),
                "--out_file",
                str(tmp_path / "n.txt"),
            ],  # fmt: skip
        )
        assert str(EXPECTED_JOBS) in result.output

    def test_materialises_the_results_tree_as_a_side_effect(self, config_file, tmp_path):
        CliRunner().invoke(
            compute_n_cli,
            [
                "--exp_config_file",
                str(config_file),
                "--root_dir",
                str(tmp_path),
                "--out_file",
                str(tmp_path / "n.txt"),
            ],  # fmt: skip
        )
        # Documented rather than desired: OutputFolder writes state.unknown on construction, so
        # merely counting creates one folder per job.
        results = tmp_path / EXPERIMENT / "results"
        assert results.is_dir()
        assert len([p for p in results.iterdir() if p.is_dir()]) == EXPECTED_JOBS

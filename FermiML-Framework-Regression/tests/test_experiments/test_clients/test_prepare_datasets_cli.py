import pytest
import yaml
from click.testing import CliRunner

from fermiml_framework_regression.experiments.clients.prepare_datasets_cli import prepare_datasets_cli


@pytest.fixture
def multi_variant_config(tmp_path):
    """A config whose one dataset is swept over two option sets."""
    path = tmp_path / "multi.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "experiment-name": "prepare_test",
                "datasets": {
                    "cv-splits": 2,
                    "splits-seed": 0,
                    "val-size": 0.25,
                    "datasets": [
                        {
                            "name": "magnetisation",
                            "extra-options": {},
                            "hyperparameters": {
                                "n_qubits": [4],
                                "n_samples": [16, 24],
                                "n_test_samples": [8],
                            },
                        }
                    ],
                },
                "models": [
                    {
                        "name": "ClassicalGP",
                        "extra-options": {},
                        "hyperparameters": {"kernel_length_scale": [1.0]},
                        "slurm-resources": {"mem_mb": 4096, "runtime_min": 30},
                    }
                ],
            }
        )
    )
    return path


class TestPrepareDatasetsCli:
    """Warms the dataset cache on a login node, before anything is submitted.

    Compute nodes have no outbound internet, so a dataset that downloads has to be materialised
    in advance into the directory the Snakefile binds as ``/app/datasets``. This runs inside the
    Apptainer image, since the cluster virtualenv holds neither scikit-learn nor this package.
    """

    def test_prepares_one_variant_per_dataset_by_default(self, config_file, tmp_path):
        result = CliRunner().invoke(
            prepare_datasets_cli,
            ["--exp_config_file", str(config_file), "--cache_dir", str(tmp_path / "cache")],
        )
        assert result.exit_code == 0, result.output
        assert "1 dataset variant(s)" in result.output
        assert "ok   magnetisation" in result.output

    def test_creates_the_cache_directory(self, config_file, tmp_path):
        cache = tmp_path / "nested" / "cache"
        result = CliRunner().invoke(
            prepare_datasets_cli, ["--exp_config_file", str(config_file), "--cache_dir", str(cache)]
        )
        assert result.exit_code == 0, result.output
        assert cache.is_dir()

    def test_points_sklearn_at_the_cache_directory(self, config_file, tmp_path, monkeypatch):
        # The whole point: without this the fetch lands in $HOME and the bound directory the
        # compute nodes see stays empty.
        import os

        cache = tmp_path / "cache"
        monkeypatch.delenv("SCIKIT_LEARN_DATA", raising=False)
        CliRunner().invoke(prepare_datasets_cli, ["--exp_config_file", str(config_file), "--cache_dir", str(cache)])
        assert os.environ["SCIKIT_LEARN_DATA"] == str(cache.resolve())

    def test_all_configs_covers_every_option_set(self, multi_variant_config, tmp_path):
        result = CliRunner().invoke(
            prepare_datasets_cli,
            ["--exp_config_file", str(multi_variant_config), "--cache_dir", str(tmp_path / "cache"), "--all_configs"],
        )
        assert result.exit_code == 0, result.output
        assert "2 dataset variant(s)" in result.output

    def test_default_collapses_variants_of_one_dataset(self, multi_variant_config, tmp_path):
        result = CliRunner().invoke(
            prepare_datasets_cli,
            ["--exp_config_file", str(multi_variant_config), "--cache_dir", str(tmp_path / "cache")],
        )
        assert result.exit_code == 0, result.output
        # One instance is enough to populate a download cache; the cache is keyed by the source,
        # not by the subsample a job takes.
        assert "1 dataset variant(s)" in result.output

    def test_reports_shapes_so_the_run_is_verifiable(self, config_file, tmp_path):
        result = CliRunner().invoke(
            prepare_datasets_cli,
            ["--exp_config_file", str(config_file), "--cache_dir", str(tmp_path / "cache")],
        )
        assert "train=" in result.output and "test=" in result.output

    def test_requires_a_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EXP_CONFIG_FILE", raising=False)
        result = CliRunner().invoke(prepare_datasets_cli, ["--cache_dir", str(tmp_path / "cache")])
        assert result.exit_code != 0
        assert "Provide --exp_config_file" in result.output

    def test_an_unknown_dataset_fails_loudly(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "experiment-name": "bad",
                    "datasets": {
                        "cv-splits": 1,
                        "datasets": [{"name": "not-a-dataset", "extra-options": {}, "hyperparameters": {}}],
                    },
                    "models": [{"name": "ClassicalGP", "extra-options": {}, "hyperparameters": {}}],
                }
            )
        )
        result = CliRunner().invoke(
            prepare_datasets_cli, ["--exp_config_file", str(path), "--cache_dir", str(tmp_path / "cache")]
        )
        # Better to fail here, on a login node, than to discover it on 10920 compute nodes.
        assert result.exit_code != 0
        assert "not a known dataset" in result.output

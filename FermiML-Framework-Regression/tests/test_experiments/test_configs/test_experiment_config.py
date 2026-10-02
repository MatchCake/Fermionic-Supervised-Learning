from importlib.metadata import version

import pytest
import yaml

from fermiml_framework_regression.experiments.configs.datasets_config import DatasetsConfig, DatasetsConfigItem
from fermiml_framework_regression.experiments.configs.experiment_config import ExperimentConfig
from fermiml_framework_regression.experiments.configs.model_hps_config import ModelHpsConfig
from fermiml_framework_regression.experiments.configs.single_experiment_config import SingleExperimentConfig


@pytest.fixture
def experiment_config():
    """A grid whose expansion is easy to count by hand: 2 datasets x 2 splits x 3 models."""
    return ExperimentConfig(
        experiment_name="expansion_test",
        datasets=DatasetsConfig(
            datasets=[
                DatasetsConfigItem(name="friedman", hyperparameters={"n_samples": [50, 100]}),
                DatasetsConfigItem(name="magnetisation", hyperparameters={"n_qubits": [4]}),
            ],
            cv_splits=2,
            val_size=0.25,
        ),
        models=[
            ModelHpsConfig(name="ClassicalGP", hyperparameters={"kernel_length_scale": [0.1, 1.0]}),
            ModelHpsConfig(name="ClassicalSVR", hyperparameters={"svr_c": [1.0]}),
        ],
    )


class TestGridExpansion:
    """The cartesian product that decides what actually runs.

    Everything downstream is sized by this: the number of SLURM jobs, the walltime of the
    sweep, and which cells the paper can report. An off-by-one in the expansion is not a
    crash, it is a quietly incomplete benchmark.
    """

    def test_dataset_configs_are_the_product_of_options_and_splits(self, experiment_config):
        # friedman has 2 n_samples values, magnetisation 1, and both run over 2 CV splits.
        assert len(experiment_config.datasets.to_dataset_configs()) == (2 + 1) * 2

    def test_dataset_hyperparameters_land_in_extra_options(self, experiment_config):
        configs = experiment_config.datasets.to_dataset_configs()
        friedman = [c for c in configs if c.dataset_name == "friedman"]
        assert {c.extra_options["n_samples"] for c in friedman} == {50, 100}

    def test_every_split_id_is_generated(self, experiment_config):
        configs = experiment_config.datasets.to_dataset_configs()
        assert {c.split_id for c in configs} == {0, 1}

    def test_shared_cv_settings_propagate(self, experiment_config):
        for config in experiment_config.datasets.to_dataset_configs():
            assert config.cv_splits == 2
            assert config.val_size == 0.25

    def test_model_configs_are_the_product_of_hyperparameters(self, experiment_config):
        assert len(experiment_config.models[0].to_model_configs()) == 2
        assert len(experiment_config.models[1].to_model_configs()) == 1

    def test_model_configs_carry_their_resources(self, experiment_config):
        for model_config in experiment_config.models[0].to_model_configs():
            assert "mem_mb" in model_config.slurm_resources

    def test_single_experiment_configs_are_the_full_product(self, experiment_config):
        assert len(experiment_config.to_single_experiment_configs()) == 6 * 3

    def test_every_expanded_job_has_a_unique_hash(self, experiment_config):
        configs = experiment_config.to_single_experiment_configs()
        # A collision would silently merge two cells into one result folder.
        assert len({c.hash for c in configs}) == len(configs)

    def test_expansion_is_stable_across_calls(self, experiment_config):
        first = [c.hash for c in experiment_config.to_single_experiment_configs()]
        second = [c.hash for c in experiment_config.to_single_experiment_configs()]
        assert first == second


class TestSingleExperimentYamls:
    """Per-job YAMLs, one per hash, used to inspect or re-run a single cell."""

    def test_writes_one_yaml_per_job(self, experiment_config, tmp_path):
        paths = experiment_config.to_single_experiment_yamls(tmp_path)
        assert len(paths) == len(experiment_config.to_single_experiment_configs())
        assert all(path.is_file() for path in paths)

    def test_each_yaml_is_named_for_its_hash_and_round_trips(self, experiment_config, tmp_path):
        for path in experiment_config.to_single_experiment_yamls(tmp_path):
            restored = SingleExperimentConfig.from_yaml_file(path)
            assert restored.hash == path.stem

    def test_written_under_the_experiment_name(self, experiment_config, tmp_path):
        paths = experiment_config.to_single_experiment_yamls(tmp_path)
        assert paths[0].parent == tmp_path / "expansion_test" / "configs"

    def test_rm_existing_clears_a_stale_directory(self, experiment_config, tmp_path):
        output_dir = tmp_path / "expansion_test" / "configs"
        experiment_config.to_single_experiment_yamls(tmp_path)
        stale = output_dir / "stale.yaml"
        stale.write_text("{}")
        experiment_config.to_single_experiment_yamls(tmp_path, rm_existing=True)
        # A hash left over from a previous grid must not survive into the new one.
        assert not stale.exists()

    def test_rm_existing_false_keeps_other_files(self, experiment_config, tmp_path):
        experiment_config.to_single_experiment_yamls(tmp_path)
        keep = tmp_path / "expansion_test" / "configs" / "keep.yaml"
        keep.write_text("{}")
        experiment_config.to_single_experiment_yamls(tmp_path, rm_existing=False)
        assert keep.exists()


class TestExperimentConfigRoundTrip:
    def test_yaml_round_trip_preserves_the_grid(self, experiment_config, tmp_path):
        path = tmp_path / "config.yaml"
        experiment_config.to_yaml_file(path)
        restored = ExperimentConfig.from_yaml_file(path)
        assert restored.experiment_name == experiment_config.experiment_name
        assert [c.hash for c in restored.to_single_experiment_configs()] == [
            c.hash for c in experiment_config.to_single_experiment_configs()
        ]

    def test_the_shipped_config_expands(self):
        # The file the cluster actually runs; a syntax or key error here is a failed sweep.
        config = ExperimentConfig.from_yaml_file("data/configs/config.yaml")
        configs = config.to_single_experiment_configs()
        assert config.experiment_name.startswith("gp_reg_benchmark")
        assert len(configs) == len({c.hash for c in configs})
        assert all(c.model.hyperparameters.get("n_qubits", -1) == -1 for c in configs)

    def test_the_experiment_name_carries_the_installed_version(self):
        """Bumping the package without bumping the tree name would reuse the previous run.

        Everything downstream keys on the experiment name -- results, configs, logs, the
        Snakefile, job_status.py -- so the version in it is what keeps two code versions from
        writing into one tree. It is a hand-maintained string in the YAML, which is exactly the
        kind of thing that gets forgotten, so it is pinned here rather than trusted.
        """
        config = ExperimentConfig.from_yaml_file("data/configs/config.yaml")
        assert config.experiment_name == f"gp_reg_benchmark-{version('FermiML-Framework-Regression')}", (
            "data/configs/config.yaml experiment-name does not match the installed package version; "
            "bump both together or the new run lands in the previous version's results tree"
        )

    def test_the_workflow_version_matches_the_package(self):
        """workflow/config.yaml carries the version the container image is tagged with.

        It names the .sif the cluster runs, so a mismatch means the sweep executes code other
        than the one this checkout describes.
        """
        payload = yaml.safe_load(open("workflow/config.yaml"))
        assert str(payload["version"]) == version("FermiML-Framework-Regression")

    def test_the_shipped_config_is_valid_yaml(self):
        payload = yaml.safe_load(open("data/configs/config.yaml"))
        assert payload["datasets"]["cv-splits"] == 20

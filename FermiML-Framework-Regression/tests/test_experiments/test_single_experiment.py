import json

import pytest

from fermiml_framework_regression.experiments.configs.dataset_config import DatasetConfig
from fermiml_framework_regression.experiments.configs.model_config import ModelConfig
from fermiml_framework_regression.experiments.configs.single_experiment_config import SingleExperimentConfig
from fermiml_framework_regression.experiments.single_experiment import SingleExperiment

PHASES = ["train", "val", "test"]
BASE_METRICS = ["mse", "rmse", "r2", "mae"]


def _config(model_name: str = "AngleNIFGP", **hyperparameters) -> SingleExperimentConfig:
    """A single job on the cheapest cell in the grid.

    :param model_name: Model to run.
    :param hyperparameters: Model hyperparameters.
    :return: The configuration.
    :rtype: SingleExperimentConfig
    """
    return SingleExperimentConfig(
        dataset=DatasetConfig(
            dataset_name="magnetisation",
            split_id=0,
            cv_splits=4,
            val_size=0.25,
            extra_options={"n_qubits": 4, "n_samples": 32, "n_test_samples": 12},
        ),
        model=ModelConfig(name=model_name, hyperparameters=dict(hyperparameters)),
    )


def _run(config: SingleExperimentConfig, root) -> dict:
    """Run one job and return the payload it wrote.

    :param config: The configuration to run.
    :param root: Root output folder.
    :return: The ``data`` block of ``run_output.out.json``.
    :rtype: dict
    """
    experiment = SingleExperiment(config, root_output_folder=root, output_folder=config.hash, log_func=lambda *_: None)
    assert experiment.run() == 0
    payload = json.loads((root / config.hash / "run_output.out.json").read_text())
    return payload["data"]


class TestSingleExperimentOutput:
    """What a finished job leaves on disk.

    This is the contract every figure and the whole Snakemake workflow depend on, and it is
    expensive to discover a gap in it after 10920 jobs have run: anything not recorded here
    cannot be recovered without recomputing the sweep.
    """

    @pytest.fixture(scope="class")
    def payload(self, tmp_path_factory):
        root = tmp_path_factory.mktemp("results")
        return _run(_config(n_qubits=-1, sector_m=2, alpha=1e-4), root)

    def test_writes_the_finished_marker(self, tmp_path):
        config = _config(n_qubits=-1, sector_m=2)
        _run(config, tmp_path)
        # state.finished is what Snakemake declares as the rule's output.
        assert (tmp_path / config.hash / "state.finished").is_file()
        assert (tmp_path / config.hash / "config.yaml").is_file()

    @pytest.mark.parametrize("phase", PHASES)
    @pytest.mark.parametrize("metric", BASE_METRICS)
    def test_records_every_metric(self, payload, phase, metric):
        assert isinstance(payload[f"{phase}_{metric}"], float)

    @pytest.mark.parametrize("key", ["n_train", "n_val", "n_test", "n_features"])
    def test_records_split_sizes(self, payload, key):
        assert isinstance(payload[key], int)

    @pytest.mark.parametrize("key", ["test_targets", "test_predictions", "val_targets", "val_predictions"])
    def test_records_per_point_arrays(self, payload, key):
        assert isinstance(payload[key], list)
        assert payload[key]

    def test_arrays_match_their_split_sizes(self, payload):
        assert len(payload["test_predictions"]) == payload["n_test"]
        assert len(payload["test_targets"]) == payload["n_test"]
        assert len(payload["val_predictions"]) == payload["n_val"]

    def test_records_predictive_std_for_a_gaussian_process(self, payload):
        assert len(payload["test_predictive_std"]) == payload["n_test"]
        assert payload["test_predictive_std_mean"] > 0

    @pytest.mark.parametrize("key", ["resolved_n_qubits", "resolved_sector_m", "resolved_n_terms"])
    def test_records_resolved_kernel_geometry(self, payload, key):
        assert isinstance(payload[key], int)

    def test_resolved_qubits_follow_the_data_width(self, payload):
        # n_qubits=-1 means "infer from the input dimension", so this is the real check that
        # auto-detection reached the kernel rather than silently keeping a default.
        assert payload["resolved_n_qubits"] == payload["n_features"]

    @pytest.mark.parametrize("key", ["gram_diagonal_mean", "gram_offdiagonal_abs_mean", "gram_offdiagonal_abs_max"])
    def test_records_gram_statistics(self, payload, key):
        assert isinstance(payload[key], float)

    @pytest.mark.parametrize("key", ["fit_seconds", "peak_rss_mb"])
    def test_records_cost(self, payload, key):
        assert payload[key] > 0

    def test_records_provenance(self, payload):
        assert payload["version_matchcake"]
        assert payload["version_fermiml_framework_regression"]

    def test_records_the_seed(self, payload):
        assert isinstance(payload["run_seed"], int)


class TestSingleExperimentSeeding:
    def test_two_runs_of_one_config_agree(self, tmp_path):
        config = _config(n_qubits=-1, sector_m=2, alpha=1e-4)
        first = _run(config, tmp_path / "a")
        second = _run(config, tmp_path / "b")
        assert first["run_seed"] == second["run_seed"]
        assert first["test_predictions"] == second["test_predictions"]
        assert first["test_r2"] == second["test_r2"]


class TestSingleExperimentFailures:
    def test_a_model_error_lands_in_state_failed(self, tmp_path):
        # sector_m must satisfy m <= 2n-2; 2n-2 = 6 here, so 8 is rejected by the kernel.
        config = _config(n_qubits=-1, sector_m=8)
        experiment = SingleExperiment(
            config, root_output_folder=tmp_path, output_folder=config.hash, log_func=lambda *_: None
        )
        assert experiment.run() == 0
        folder = tmp_path / config.hash
        assert not (folder / "state.finished").is_file()
        failed = folder / "state.failed"
        assert failed.is_file()
        # The message is written into the state file; it is the best failure diagnostic there is.
        assert "sector_m" in failed.read_text()

    def test_an_unknown_model_raises_before_running(self, tmp_path):
        config = _config("NotAModel")
        with pytest.raises(ValueError, match="not supported"):
            SingleExperiment(config, root_output_folder=tmp_path, output_folder=config.hash, log_func=lambda *_: None)

    def test_an_unknown_dataset_raises_before_running(self, tmp_path):
        config = _config(n_qubits=-1, sector_m=2)
        config.dataset.dataset_name = "not-a-dataset"
        with pytest.raises(ValueError, match="not supported"):
            SingleExperiment(config, root_output_folder=tmp_path, output_folder=config.hash, log_func=lambda *_: None)


class TestSingleExperimentWithoutAKernel:
    """A classical model has no kernel, and must still finish rather than fail on a diagnostic."""

    def test_classical_model_finishes_without_kernel_diagnostics(self, tmp_path):
        config = _config("ClassicalGP", kernel_length_scale=1.0)
        payload = _run(config, tmp_path)
        assert (tmp_path / config.hash / "state.finished").is_file()
        assert isinstance(payload["test_r2"], float)
        assert "resolved_n_terms" not in payload
        assert "gram_diagonal_mean" not in payload
        # ClassicalGP fits its own kernel, so what it converged to is recorded instead.
        assert "RBF" in payload["fitted_kernel"]

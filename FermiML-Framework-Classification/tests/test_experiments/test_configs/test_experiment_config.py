from pathlib import Path

import pytest

from fermiml_framework_classification.experiments.configs.dataset_config import (
    DatasetConfig,
)
from fermiml_framework_classification.experiments.configs.datasets_config import (
    DatasetsConfig,
    DatasetsConfigItem,
)
from fermiml_framework_classification.experiments.configs.experiment_config import (
    ExperimentConfig,
)
from fermiml_framework_classification.experiments.configs.model_config import (
    ModelConfig,
)
from fermiml_framework_classification.experiments.configs.model_hps_config import (
    ModelHpsConfig,
)
from fermiml_framework_classification.experiments.configs.single_experiment_config import (
    SingleExperimentConfig,
)


@pytest.fixture
def narrow_dataset():
    """A two-dimensional dataset configuration, the case that makes ``sector_m=4`` infeasible."""
    return DatasetConfig(dataset_name="downscaled-mnist", extra_options={"dimension": 2})


@pytest.fixture
def wide_dataset():
    """A twenty-dimensional dataset configuration, where every swept ``sector_m`` fits."""
    return DatasetConfig(dataset_name="hidden-manifold", extra_options={"dimensions": 20})


@pytest.fixture
def config():
    return ExperimentConfig(
        experiment_name="test_exp",
        datasets=DatasetsConfig(
            datasets=[DatasetsConfigItem(name="bars-and-stripes")],
            cv_splits=2,
        ),
        models=[ModelHpsConfig(name="IQP", hyperparameters={"svm_c": [0.1, 1]})],
    )


class TestExperimentConfig:
    def test_name_stored(self, config):
        assert config.experiment_name == "test_exp"

    def test_datasets_stored(self, config):
        assert len(config.datasets.datasets) == 1

    def test_models_stored(self, config):
        assert len(config.models) == 1

    def test_to_single_experiment_configs_returns_list(self, config):
        result = config.to_single_experiment_configs()
        assert isinstance(result, list)

    def test_to_single_experiment_configs_type(self, config):
        assert all(isinstance(c, SingleExperimentConfig) for c in config.to_single_experiment_configs())

    def test_to_single_experiment_configs_count(self, config):
        # 1 dataset × 2 cv_splits × 2 svm_c values = 4
        assert len(config.to_single_experiment_configs()) == 4

    def test_to_single_experiment_configs_unique_hashes(self, config):
        configs = config.to_single_experiment_configs()
        hashes = [c.hash for c in configs]
        assert len(hashes) == len(set(hashes))

    def test_to_single_experiment_yamls_creates_files(self, config, tmp_path):
        paths = config.to_single_experiment_yamls(tmp_path)
        assert len(paths) == 4
        assert all(p.exists() for p in paths)

    def test_to_single_experiment_yamls_output_dir(self, config, tmp_path):
        paths = config.to_single_experiment_yamls(tmp_path)
        expected_dir = tmp_path / "test_exp" / "configs"
        assert all(p.parent == expected_dir for p in paths)

    def test_to_single_experiment_yamls_rm_existing(self, config, tmp_path):
        config.to_single_experiment_yamls(tmp_path)
        paths = config.to_single_experiment_yamls(tmp_path, rm_existing=True)
        assert len(list((tmp_path / "test_exp" / "configs").iterdir())) == 4

    def test_to_single_experiment_yamls_no_rm_existing(self, config, tmp_path):
        config.to_single_experiment_yamls(tmp_path, rm_existing=False)
        paths = config.to_single_experiment_yamls(tmp_path, rm_existing=False)
        assert len(paths) == 4

    def test_model_config_is_feasible_keeps_small_pfaffian_grid(self, wide_dataset):
        # C(2*32, 4) = 635,376 is under the limit.
        model_config = ModelConfig(name="VanillaMpf", hyperparameters={"n_qubits": 32, "sector_m": 4})
        assert ExperimentConfig._model_config_is_feasible(model_config, wide_dataset)

    def test_model_config_is_feasible_drops_large_pfaffian_grid(self, wide_dataset):
        # C(2*64, 4) = 10,668,000 exceeds the limit.
        model_config = ModelConfig(name="VanillaMpf", hyperparameters={"n_qubits": 64, "sector_m": 4})
        assert not ExperimentConfig._model_config_is_feasible(model_config, wide_dataset)

    def test_model_config_is_feasible_keeps_inferred_n_qubits(self, wide_dataset):
        # n_qubits = -1 is inferred at fit time; on a 20-dimensional dataset that leaves room for m=4.
        model_config = ModelConfig(name="VanillaMpf", hyperparameters={"n_qubits": -1, "sector_m": 4})
        assert ExperimentConfig._model_config_is_feasible(model_config, wide_dataset)

    def test_model_config_is_feasible_drops_inferred_n_qubits_on_two_dimensional_data(self, narrow_dataset):
        # d=2 gives n=2 and 2n-2=2, so no non-trivial 4-subset of the Majorana indices exists and fit raises.
        model_config = ModelConfig(name="VanillaMpf", hyperparameters={"n_qubits": -1, "sector_m": 4})
        assert not ExperimentConfig._model_config_is_feasible(model_config, narrow_dataset)

    def test_model_config_is_feasible_keeps_sector_two_on_two_dimensional_data(self, narrow_dataset):
        model_config = ModelConfig(name="VanillaMpf", hyperparameters={"n_qubits": -1, "sector_m": 2})
        assert ExperimentConfig._model_config_is_feasible(model_config, narrow_dataset)

    def test_model_config_is_feasible_checks_every_spanned_sector(self, wide_dataset):
        model_config = ModelConfig(name="VanillaMpf+", hyperparameters={"n_qubits": -1, "sectors": [2, 3]})
        assert ExperimentConfig._model_config_is_feasible(model_config, wide_dataset)

    def test_model_config_is_feasible_keeps_odd_sector_on_two_dimensional_data(self, narrow_dataset):
        # An odd order is closed with the parity index, so it needs only m <= 2n = 4, unlike the even bound
        # of 2n-2 = 2 that rejects sector_m=4 here. Every configuration therefore stays populated.
        model_config = ModelConfig(name="VanillaMpf+", hyperparameters={"n_qubits": -1, "sectors": [2, 3]})
        assert ExperimentConfig._model_config_is_feasible(model_config, narrow_dataset)

    def test_model_config_is_feasible_drops_an_odd_sector_beyond_the_index_count(self, narrow_dataset):
        model_config = ModelConfig(name="VanillaMpf+", hyperparameters={"n_qubits": -1, "sectors": [5]})
        assert not ExperimentConfig._model_config_is_feasible(model_config, narrow_dataset)

    def test_model_config_is_feasible_sums_dimensions_across_sectors(self, wide_dataset):
        # Each order fits on its own, but their sum blows the feature ceiling.
        model_config = ModelConfig(name="VanillaMpf+", hyperparameters={"n_qubits": 64, "sectors": [2, 4]})
        assert not ExperimentConfig._model_config_is_feasible(model_config, wide_dataset)

    def test_model_config_is_feasible_prefers_sectors_over_sector_m(self, narrow_dataset):
        # sectors wins when both are present, so a leftover sector_m cannot veto a feasible spec.
        model_config = ModelConfig(name="VanillaMpf+", hyperparameters={"n_qubits": -1, "sector_m": 4, "sectors": [2]})
        assert ExperimentConfig._model_config_is_feasible(model_config, narrow_dataset)

    @pytest.mark.parametrize("name", ["PQC", "fPQC"])
    @pytest.mark.parametrize(
        "n_qubits, dimension, expected",
        [
            (-1, 10, True),
            (4, 10, True),
            (8, 10, True),
            (10, 10, False),
            (16, 10, False),
            (4, 4, False),
            (4, 3, False),
            (4, 5, True),
            (-1, 2, True),
        ],
    )
    def test_model_config_is_feasible_drops_repeated_pqc_circuits(self, name, n_qubits, dimension, expected):
        model = ModelConfig(name=name, hyperparameters={"n_qubits": n_qubits, "svm_c": 1})
        dataset = DatasetConfig(dataset_name="linearly-separable", extra_options={"dimension": dimension})
        assert ExperimentConfig._model_config_is_feasible(model, dataset) is expected

    @pytest.mark.parametrize("name", ["PQC", "fPQC"])
    @pytest.mark.parametrize(
        "rule, dimension, expected",
        [
            ("full", 2, True),
            ("half", 2, False),
            ("quarter", 2, False),
            ("half", 4, True),
            ("quarter", 4, False),
            ("half", 6, True),
            ("quarter", 6, True),
            ("half", 20, True),
            ("quarter", 20, True),
            ("eighth", 20, True),
            ("eighth", 16, True),
            ("eighth", 10, True),
            ("eighth", 4, False),
            ("eighth", 6, False),
        ],
    )
    def test_model_config_is_feasible_drops_repeated_pqc_rules(self, name, rule, dimension, expected):
        model = ModelConfig(name=name, hyperparameters={"n_qubits": rule, "svm_c": 1})
        dataset = DatasetConfig(dataset_name="downscaled-mnist", extra_options={"dimension": dimension})
        assert ExperimentConfig._model_config_is_feasible(model, dataset) is expected

    @pytest.mark.parametrize("dimension", [2, 3, 20])
    def test_model_config_is_feasible_keeps_doubled_fpqc(self, dimension):
        model = ModelConfig(name="fPQC", hyperparameters={"n_qubits": "double", "svm_c": 1})
        dataset = DatasetConfig(dataset_name="downscaled-mnist", extra_options={"dimension": dimension})
        assert ExperimentConfig._model_config_is_feasible(model, dataset) is True

    def test_model_config_is_feasible_drops_doubled_pqc(self):
        # Only the matchgate model has a doubled encoding; the unrestricted one would reject the rule at fit time.
        model = ModelConfig(name="PQC", hyperparameters={"n_qubits": "double", "svm_c": 1})
        dataset = DatasetConfig(dataset_name="downscaled-mnist", extra_options={"dimension": 10})
        assert ExperimentConfig._model_config_is_feasible(model, dataset) is False

    def test_model_config_is_feasible_keeps_non_pfaffian_model(self, narrow_dataset):
        model_config = ModelConfig(name="IQP", hyperparameters={"svm_c": 1, "repeats": 5})
        assert ExperimentConfig._model_config_is_feasible(model_config, narrow_dataset)

    def test_model_config_is_feasible_keeps_unknown_dataset(self):
        # An unrecognised dataset name has no declared dimension, so the decision is left to the job.
        model_config = ModelConfig(name="VanillaMpf", hyperparameters={"n_qubits": -1, "sector_m": 4})
        unknown = DatasetConfig(dataset_name="not-a-dataset", extra_options={})
        assert ExperimentConfig._model_config_is_feasible(model_config, unknown)

    def test_dataset_feature_dimension_reads_the_dataset_class(self, narrow_dataset, wide_dataset):
        assert ExperimentConfig._dataset_feature_dimension(narrow_dataset) == 2
        assert ExperimentConfig._dataset_feature_dimension(wide_dataset) == 20

    def test_dataset_feature_dimension_unknown_name(self):
        assert ExperimentConfig._dataset_feature_dimension(DatasetConfig(dataset_name="nope")) is None

    def test_to_single_experiment_configs_drops_infeasible_combos(self):
        config = ExperimentConfig(
            experiment_name="test_exp",
            datasets=DatasetsConfig(
                datasets=[DatasetsConfigItem(name="bars-and-stripes")],
                cv_splits=2,
            ),
            models=[
                ModelHpsConfig(
                    name="VanillaMpf",
                    hyperparameters={"n_qubits": [32, 64], "sector_m": [4], "svm_c": [1]},
                )
            ],
        )
        single_configs = config.to_single_experiment_configs()
        # 2 dataset splits × only the feasible n_qubits=32 combo (n_qubits=64 dropped).
        assert len(single_configs) == 2
        assert all(c.model.hyperparameters["n_qubits"] == 32 for c in single_configs)

    def test_shipped_config_has_no_infeasible_pfaffian_combos(self):
        # Regression guard for the cluster OOM fix: the production grid must never
        # pair sector_m>=4 with an explicit n_qubits. Those combinations exhaust the
        # Pfaffian feature map and get OOM-killed on the cluster. Only the
        # data-inferred n_qubits=-1 may carry sector_m=4.
        config_path = Path(__file__).parents[3] / "data" / "configs" / "config_20splits.yaml"
        exp_config = ExperimentConfig.from_yaml_file(config_path)
        offenders = [
            model_config
            for model_hps in exp_config.models
            for model_config in model_hps.to_model_configs()
            if model_config.hyperparameters.get("sector_m", 0) >= 4
            and model_config.hyperparameters.get("n_qubits", -1) >= 0
        ]
        assert offenders == []

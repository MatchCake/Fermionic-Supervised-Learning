import pytest

from fermiml_framework_classification.experiments.configs.dataset_config import (
    DatasetConfig,
)
from fermiml_framework_classification.experiments.configs.model_config import (
    ModelConfig,
)
from fermiml_framework_classification.experiments.configs.single_experiment_config import (
    SingleExperimentConfig,
)


class TestSingleExperimentConfig:
    @pytest.fixture
    def dataset_config(self):
        return DatasetConfig(dataset_name="bars-and-stripes")

    @pytest.fixture
    def model_config(self):
        return ModelConfig(name="IQP", hyperparameters={})

    @pytest.fixture
    def config(self, dataset_config, model_config):
        return SingleExperimentConfig(dataset=dataset_config, model=model_config)

    # --- construction ---

    def test_stores_dataset(self, config, dataset_config):
        assert config.dataset == dataset_config

    def test_stores_model(self, config, model_config):
        assert config.model == model_config

    # --- to_dict ---

    def test_to_dict_returns_dict(self, config):
        assert isinstance(config.to_dict(), dict)

    def test_to_dict_contains_dataset_key(self, config):
        assert "dataset" in config.to_dict()

    def test_to_dict_contains_model_key(self, config):
        assert "model" in config.to_dict()

    def test_to_dict_dataset_name(self, config):
        assert config.to_dict()["dataset"]["dataset-name"] == "bars-and-stripes"

    def test_to_dict_model_name(self, config):
        assert config.to_dict()["model"]["name"] == "IQP"

    # --- hash ---

    def test_hash_returns_string(self, config):
        assert isinstance(config.hash, str)

    def test_hash_is_deterministic(self, config):
        assert config.hash == config.to_hash()

    def test_different_configs_have_different_hashes(self, model_config):
        c1 = SingleExperimentConfig(dataset=DatasetConfig(dataset_name="a"), model=model_config)
        c2 = SingleExperimentConfig(dataset=DatasetConfig(dataset_name="b"), model=model_config)
        assert c1.hash != c2.hash

    def test_equal_configs_have_equal_hashes(self, model_config):
        c1 = SingleExperimentConfig(dataset=DatasetConfig(dataset_name="x"), model=model_config)
        c2 = SingleExperimentConfig(dataset=DatasetConfig(dataset_name="x"), model=model_config)
        assert c1.hash == c2.hash

    def test_hash_ignores_slurm_resources(self, dataset_config):
        cheap = SingleExperimentConfig(
            dataset=dataset_config,
            model=ModelConfig(
                name="IQP",
                hyperparameters={},
                slurm_resources={"mem_mb": 1024, "runtime_min": 10},
            ),
        )
        expensive = SingleExperimentConfig(
            dataset=dataset_config,
            model=ModelConfig(
                name="IQP",
                hyperparameters={},
                slurm_resources={"mem_mb": 131072, "runtime_min": 4320},
            ),
        )
        assert cheap.hash == expensive.hash

    def test_to_dict_serialises_slurm_resources_under_dash_key(self, config):
        # The dash spelling is what to_hash must strip; an attribute-spelled pop
        # silently matches nothing and leaks the resources into the digest.
        assert "slurm-resources" in config.to_dict()["model"]

    # --- repr ---

    def test_repr_contains_class_name(self, config):
        assert "SingleExperimentConfig" in repr(config)

    # --- YAML round-trip ---

    def test_to_yaml_returns_string(self, config):
        assert isinstance(config.to_yaml(), str)

    def test_yaml_round_trip(self, config):
        restored = SingleExperimentConfig.from_yaml(config.to_yaml())
        assert restored.dataset == config.dataset
        assert restored.model == config.model

    def test_yaml_file_round_trip(self, config, tmp_path):
        path = tmp_path / "config.yaml"
        config.to_yaml_file(path)
        restored = SingleExperimentConfig.from_yaml_file(path)
        assert restored.dataset == config.dataset
        assert restored.model == config.model

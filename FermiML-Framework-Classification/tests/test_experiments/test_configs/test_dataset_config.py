import pytest

from fermiml_framework_classification.experiments.configs.dataset_config import (
    DatasetConfig,
)


class TestDatasetConfig:
    @pytest.fixture
    def config(self):
        return DatasetConfig(dataset_name="bars-and-stripes")

    # --- defaults ---

    def test_default_split_id(self, config):
        assert config.split_id == 0

    def test_default_splits_seed(self, config):
        assert config.splits_seed == 0

    def test_default_cv_splits(self, config):
        assert config.cv_splits == 5

    def test_default_val_size(self, config):
        assert config.val_size == 0.2

    def test_default_extra_options(self, config):
        assert config.extra_options == {}

    def test_extra_options_independent_per_instance(self):
        a = DatasetConfig(dataset_name="a")
        b = DatasetConfig(dataset_name="b")
        a.extra_options["key"] = "value"
        assert "key" not in b.extra_options

    # --- custom values ---

    def test_custom_dataset_name(self):
        config = DatasetConfig(dataset_name="mnist")
        assert config.dataset_name == "mnist"

    def test_custom_split_id(self):
        config = DatasetConfig(dataset_name="x", split_id=3)
        assert config.split_id == 3

    def test_custom_splits_seed(self):
        config = DatasetConfig(dataset_name="x", splits_seed=42)
        assert config.splits_seed == 42

    def test_custom_cv_splits(self):
        config = DatasetConfig(dataset_name="x", cv_splits=10)
        assert config.cv_splits == 10

    def test_custom_val_size(self):
        config = DatasetConfig(dataset_name="x", val_size=0.1)
        assert config.val_size == 0.1

    def test_custom_extra_options(self):
        config = DatasetConfig(dataset_name="x", extra_options={"image_size": 8})
        assert config.extra_options == {"image_size": 8}

    # --- YAML round-trip ---

    def test_to_yaml_returns_string(self, config):
        assert isinstance(config.to_yaml(), str)

    def test_yaml_round_trip(self, config):
        restored = DatasetConfig.from_yaml(config.to_yaml())
        assert restored == config

    def test_yaml_contains_dataset_name(self, config):
        assert "bars-and-stripes" in config.to_yaml()

    def test_yaml_round_trip_custom_values(self):
        config = DatasetConfig(
            dataset_name="mnist",
            split_id=2,
            splits_seed=7,
            cv_splits=10,
            val_size=0.15,
            extra_options={"image_size": 16},
        )
        restored = DatasetConfig.from_yaml(config.to_yaml())
        assert restored == config

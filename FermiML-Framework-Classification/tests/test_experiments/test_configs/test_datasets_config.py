import pytest

from fermiml_framework_classification.experiments.configs.dataset_config import (
    DatasetConfig,
)
from fermiml_framework_classification.experiments.configs.datasets_config import (
    DatasetsConfig,
    DatasetsConfigItem,
)


class TestDatasetsConfigItem:
    def test_name_stored(self):
        item = DatasetsConfigItem(name="bars-and-stripes")
        assert item.name == "bars-and-stripes"

    def test_extra_options_default_empty(self):
        assert DatasetsConfigItem(name="bars-and-stripes").extra_options == {}

    def test_extra_options_stored(self):
        item = DatasetsConfigItem(name="bars-and-stripes", extra_options={"image_size": 4})
        assert item.extra_options == {"image_size": 4}

    def test_hyperparameters_default_empty(self):
        assert DatasetsConfigItem(name="bars-and-stripes").hyperparameters == {}

    def test_hyperparameters_stored(self):
        item = DatasetsConfigItem(name="bars-and-stripes", hyperparameters={"image_size": [4, 8]})
        assert item.hyperparameters == {"image_size": [4, 8]}


class TestDatasetsConfig:
    @pytest.fixture
    def config(self):
        return DatasetsConfig(
            datasets=[
                DatasetsConfigItem(name="bars-and-stripes"),
                DatasetsConfigItem(name="hidden-manifold", extra_options={"n_manifolds": 4}),
            ],
            splits_seed=42,
            cv_splits=3,
            val_size=0.2,
        )

    def test_datasets_stored(self, config):
        assert len(config.datasets) == 2

    def test_splits_seed_stored(self, config):
        assert config.splits_seed == 42

    def test_cv_splits_stored(self, config):
        assert config.cv_splits == 3

    def test_val_size_stored(self, config):
        assert config.val_size == 0.2

    def test_to_dataset_configs_returns_dataset_configs(self, config):
        result = config.to_dataset_configs()
        assert all(isinstance(c, DatasetConfig) for c in result)

    def test_to_dataset_configs_count(self, config):
        # 2 datasets × 3 cv_splits × 1 hp_combo (no HPs) = 6
        assert len(config.to_dataset_configs()) == 6

    def test_to_dataset_configs_count_with_hyperparameters(self):
        # 1 dataset × 2 cv_splits × 2 image_size × 3 n_train = 12
        config = DatasetsConfig(
            datasets=[
                DatasetsConfigItem(
                    name="bars-and-stripes", hyperparameters={"image_size": [4, 8], "n_train": [10, 20, 30]}
                )
            ],
            cv_splits=2,
        )
        assert len(config.to_dataset_configs()) == 12

    def test_to_dataset_configs_hp_values_in_extra_options(self):
        config = DatasetsConfig(
            datasets=[DatasetsConfigItem(name="bars-and-stripes", hyperparameters={"image_size": [4, 8]})],
            cv_splits=1,
        )
        image_sizes = {c.extra_options["image_size"] for c in config.to_dataset_configs()}
        assert image_sizes == {4, 8}

    def test_to_dataset_configs_merges_extra_options_and_hyperparameters(self):
        config = DatasetsConfig(
            datasets=[
                DatasetsConfigItem(
                    name="hidden-manifold",
                    extra_options={"n_manifolds": 4},
                    hyperparameters={"n_train": [10, 20]},
                )
            ],
            cv_splits=1,
        )
        results = config.to_dataset_configs()
        assert all(c.extra_options["n_manifolds"] == 4 for c in results)
        assert {c.extra_options["n_train"] for c in results} == {10, 20}

    def test_to_dataset_configs_no_hyperparameters_produces_one_per_split(self):
        config = DatasetsConfig(
            datasets=[DatasetsConfigItem(name="bars-and-stripes")],
            cv_splits=3,
        )
        assert len(config.to_dataset_configs()) == 3

    def test_to_dataset_configs_split_ids(self, config):
        split_ids = {c.split_id for c in config.to_dataset_configs()}
        assert split_ids == {0, 1, 2}

    def test_to_dataset_configs_dataset_names(self, config):
        names = {c.dataset_name for c in config.to_dataset_configs()}
        assert names == {"bars-and-stripes", "hidden-manifold"}

    def test_to_dataset_configs_propagates_splits_seed(self, config):
        assert all(c.splits_seed == 42 for c in config.to_dataset_configs())

    def test_to_dataset_configs_propagates_cv_splits(self, config):
        assert all(c.cv_splits == 3 for c in config.to_dataset_configs())

    def test_to_dataset_configs_propagates_val_size(self, config):
        assert all(c.val_size == 0.2 for c in config.to_dataset_configs())

    def test_to_dataset_configs_propagates_extra_options(self, config):
        hm_configs = [c for c in config.to_dataset_configs() if c.dataset_name == "hidden-manifold"]
        assert all(c.extra_options == {"n_manifolds": 4} for c in hm_configs)

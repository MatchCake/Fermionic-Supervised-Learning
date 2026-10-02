import pytest

from fermihamml.experiments.configs.dataset_config import DatasetConfig


class TestDatasetConfig:
    @pytest.fixture
    def dataset_config_instance(self):
        return DatasetConfig(
            dataset_name="nif_ising_cls",
            split_id=0,
            seed=0,
            batch_size=32,
        )

    def test_attrs(self, dataset_config_instance):
        assert dataset_config_instance.dataset_name == "nif_ising_cls"
        assert dataset_config_instance.split_id == 0
        assert dataset_config_instance.seed == 0
        assert dataset_config_instance.batch_size == 32

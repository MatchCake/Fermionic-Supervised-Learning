import pytest

from fermihamml.experiments.configs.dataset_config import DatasetConfig
from fermihamml.experiments.configs.model_config import ModelConfig
from fermihamml.experiments.configs.opt_config import OptConfig
from fermihamml.experiments.configs.single_experiment_config import (
    SingleExperimentConfig,
)


class TestSingleExperimentConfig:
    @pytest.fixture
    def config_instance(self):
        dataset_config = DatasetConfig(
            dataset_name="nif_ising_cls",
            split_id=0,
            seed=0,
            batch_size=32,
        )
        opt_config = OptConfig(
            automl_iterations=10,
            inner_max_epochs=5,
            inner_max_time="00:00:10:00",
            max_epochs=10,
            max_time="00:00:20:00",
            monitor="val_loss",
            automl_overwrite_fit=True,
            accelerator="cpu",
            extra_options={"foo": "bar"},
        )
        return SingleExperimentConfig(
            model=ModelConfig(
                name="test_model",
                extra_options={"foo": "bar"},
            ),
            dataset_config=dataset_config,
            opt_config=opt_config,
        )

    def test_attrs(self, config_instance):
        assert config_instance.model.name == "test_model"
        assert config_instance.model.extra_options == {"foo": "bar"}

        assert config_instance.dataset_config.dataset_name == "nif_ising_cls"
        assert config_instance.dataset_config.split_id == 0
        assert config_instance.dataset_config.seed == 0
        assert config_instance.dataset_config.batch_size == 32

        assert config_instance.opt_config.automl_iterations == 10
        assert config_instance.opt_config.inner_max_epochs == 5
        assert config_instance.opt_config.inner_max_time == "00:00:10:00"
        assert config_instance.opt_config.max_epochs == 10
        assert config_instance.opt_config.max_time == "00:00:20:00"
        assert config_instance.opt_config.monitor == "val_loss"
        assert config_instance.opt_config.automl_overwrite_fit is True
        assert config_instance.opt_config.accelerator == "cpu"
        assert config_instance.opt_config.extra_options == {"foo": "bar"}

    def test_to_dict(self, config_instance):
        config_dict = config_instance.to_dict()
        assert config_dict["model"] == {"name": "test_model", "extra_options": {"foo": "bar"}}
        assert config_dict["dataset_config"]["dataset_name"] == "nif_ising_cls"
        assert config_dict["dataset_config"]["split_id"] == 0
        assert config_dict["dataset_config"]["seed"] == 0
        assert config_dict["dataset_config"]["batch_size"] == 32
        assert config_dict["opt_config"]["automl_iterations"] == 10
        assert config_dict["opt_config"]["inner_max_epochs"] == 5
        assert config_dict["opt_config"]["inner_max_time"] == "00:00:10:00"
        assert config_dict["opt_config"]["max_epochs"] == 10
        assert config_dict["opt_config"]["max_time"] == "00:00:20:00"
        assert config_dict["opt_config"]["monitor"] == "val_loss"
        assert config_dict["opt_config"]["automl_overwrite_fit"] is True
        assert config_dict["opt_config"]["accelerator"] == "cpu"
        assert config_dict["opt_config"]["extra_options"] == {"foo": "bar"}

    def test_to_hash(self, config_instance):
        config_hash = config_instance.to_hash()
        assert isinstance(config_hash, str)
        assert len(config_hash) > 0
        assert config_hash == config_instance.hash

import pytest

from fermiml_framework_classification.experiments.configs.model_config import (
    ModelConfig,
)
from fermiml_framework_classification.experiments.configs.model_hps_config import (
    ModelHpsConfig,
)


class TestModelHpsConfig:
    @pytest.fixture
    def config(self):
        return ModelHpsConfig(name="IQP", hyperparameters={"repeats": [1, 5], "svm_c": [0.1, 1]})

    def test_name_stored(self, config):
        assert config.name == "IQP"

    def test_hyperparameters_stored(self, config):
        assert config.hyperparameters["repeats"] == [1, 5]

    def test_extra_options_default_empty(self):
        assert ModelHpsConfig(name="IQP").extra_options == {}

    def test_to_model_configs_returns_model_configs(self, config):
        assert all(isinstance(c, ModelConfig) for c in config.to_model_configs())

    def test_to_model_configs_cartesian_product(self, config):
        assert len(config.to_model_configs()) == 4  # 2 repeats × 2 svm_c

    def test_to_model_configs_all_combinations_present(self, config):
        hp_tuples = {(c.hyperparameters["repeats"], c.hyperparameters["svm_c"]) for c in config.to_model_configs()}
        assert hp_tuples == {(1, 0.1), (1, 1), (5, 0.1), (5, 1)}

    def test_to_model_configs_single_value(self):
        result = ModelHpsConfig(name="IQP", hyperparameters={"svm_c": [1]}).to_model_configs()
        assert len(result) == 1
        assert result[0].hyperparameters == {"svm_c": 1}

    def test_to_model_configs_empty_hyperparameters(self):
        result = ModelHpsConfig(name="IQP", hyperparameters={}).to_model_configs()
        assert len(result) == 1
        assert result[0].hyperparameters == {}

    def test_to_model_configs_name_propagated(self, config):
        assert all(c.name == "IQP" for c in config.to_model_configs())

    def test_to_model_configs_extra_options_propagated(self):
        c = ModelHpsConfig(name="IQP", hyperparameters={"svm_c": [1]}, extra_options={"foo": "bar"})
        assert c.to_model_configs()[0].extra_options == {"foo": "bar"}

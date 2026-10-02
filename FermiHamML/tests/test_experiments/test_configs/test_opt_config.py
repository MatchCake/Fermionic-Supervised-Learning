import pytest

from fermihamml.experiments.configs.opt_config import OptConfig


class TestOptConfig:
    @pytest.fixture
    def config_instance(self):
        return OptConfig(
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

    def test_attrs(self, config_instance):
        assert config_instance.automl_iterations == 10
        assert config_instance.inner_max_epochs == 5
        assert config_instance.inner_max_time == "00:00:10:00"
        assert config_instance.max_epochs == 10
        assert config_instance.max_time == "00:00:20:00"
        assert config_instance.monitor == "val_loss"
        assert config_instance.automl_overwrite_fit is True
        assert config_instance.accelerator == "cpu"
        assert config_instance.extra_options == {"foo": "bar"}

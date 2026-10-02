import pytest
from dataclass_wizard import asdict

from fermiml_framework_regression.experiments.configs.dataset_config import DatasetConfig
from fermiml_framework_regression.experiments.configs.model_config import ModelConfig
from fermiml_framework_regression.experiments.configs.single_experiment_config import SingleExperimentConfig


def _config(mem_mb: int = 8192, runtime_min: int = 60, **model_hyperparameters) -> SingleExperimentConfig:
    """Build a config differing only in the fields a test varies.

    :param mem_mb: Memory request placed in ``slurm_resources``.
    :param runtime_min: Walltime request placed in ``slurm_resources``.
    :param model_hyperparameters: Model hyperparameters.
    :return: The config.
    :rtype: SingleExperimentConfig
    """
    return SingleExperimentConfig(
        dataset=DatasetConfig(dataset_name="friedman", split_id=0, cv_splits=20),
        model=ModelConfig(
            name="AngleNIFGP",
            hyperparameters=dict(model_hyperparameters) or {"sector_m": 2},
            slurm_resources={"mem_mb": mem_mb, "runtime_min": runtime_min},
        ),
    )


class TestSingleExperimentConfigHash:
    """The hash addresses the result folder, so what it does and does not cover is a contract."""

    def test_serialisation_uses_kebab_case(self):
        # The reason the exclusion needs both spellings; if this ever flips to snake_case
        # the popped-key list is what has to change, not the hash.
        assert "slurm-resources" in asdict(_config())["model"]

    def test_memory_request_does_not_change_the_hash(self):
        assert _config(mem_mb=8192).to_hash() == _config(mem_mb=131072).to_hash()

    def test_walltime_request_does_not_change_the_hash(self):
        assert _config(runtime_min=60).to_hash() == _config(runtime_min=960).to_hash()

    def test_hyperparameters_do_change_the_hash(self):
        assert _config(sector_m=2).to_hash() != _config(sector_m=4).to_hash()

    def test_dataset_does_change_the_hash(self):
        first = _config()
        second = _config()
        second.dataset.split_id = 3
        assert first.to_hash() != second.to_hash()

    def test_hash_is_64_hex_characters(self):
        digest = _config().to_hash()
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)

    def test_cached_hash_matches_to_hash(self):
        config = _config()
        assert config.hash == config.to_hash()

    @pytest.mark.parametrize("key", SingleExperimentConfig.HASH_EXCLUDED_MODEL_KEYS)
    def test_excluded_keys_are_absent_from_the_hashed_payload(self, key):
        config = _config()
        payload = config.to_dict()
        for excluded in SingleExperimentConfig.HASH_EXCLUDED_MODEL_KEYS:
            payload.get("model", {}).pop(excluded, None)
        assert key not in payload["model"]

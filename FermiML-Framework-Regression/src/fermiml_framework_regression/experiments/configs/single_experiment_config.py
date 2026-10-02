import json
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict

from dataclass_wizard import YAMLWizard, asdict
from pythonbasictools.hash_tools import hash_dict

from .dataset_config import DatasetConfig
from .model_config import ModelConfig


@dataclass
class SingleExperimentConfig(YAMLWizard):
    """Configuration for a single (dataset, model, split) experiment.

    :param dataset: Dataset configuration.
    :param model: Model configuration.
    """

    dataset: DatasetConfig
    model: ModelConfig

    def __repr__(self) -> str:
        return f"SingleExperimentConfig{json.dumps(self.to_dict(), indent=4, default=str)}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary.

        :return: Dictionary representation.
        """
        return asdict(self)

    HASH_EXCLUDED_MODEL_KEYS = ("slurm_resources", "slurm-resources")

    def to_hash(self) -> str:
        """Compute a 64-character hex hash that excludes ``slurm_resources``.

        Resource requests are deliberately outside the digest so that re-provisioning a
        job does not invalidate its cached result.

        :return: Hex string hash.
        :rtype: str
        """
        d = self.to_dict()
        model = d.get("model", {})
        for key in self.HASH_EXCLUDED_MODEL_KEYS:
            model.pop(key, None)
        return hash_dict(d)

    @cached_property
    def hash(self) -> str:
        """Cached experiment hash.

        :rtype: str
        """
        return self.to_hash()

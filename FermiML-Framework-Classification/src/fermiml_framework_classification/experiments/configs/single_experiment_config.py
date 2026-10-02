import json
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict

from dataclass_wizard import YAMLWizard, asdict
from pythonbasictools.hash_tools import hash_dict

from .dataset_config import DatasetConfig
from .model_config import ModelConfig


@dataclass
class SingleExperimentConfig(YAMLWizard):
    dataset: DatasetConfig
    model: ModelConfig

    def __repr__(self):
        return f"SingleExperimentConfig" f"{json.dumps(self.to_dict(), indent=4, default=str)}" f""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_hash(self) -> str:
        """Digest the dataset and model hyperparameters, ignoring SLURM resources.

        ``asdict`` serialises the ``slurm_resources`` field under its YAML key
        ``slurm-resources``, so popping only the attribute spelling matched
        nothing and left the resources in the digest. Every ``mem_mb`` or
        ``runtime_min`` edit then changed the hash and orphaned finished
        results. Both spellings are removed so the digest stays stable
        whichever one the serializer emits.

        :return: The 64-character hex digest of the config.
        :rtype: str
        """
        d = self.to_dict()
        model_dict = d.get("model", {})
        for resources_key in ("slurm_resources", "slurm-resources"):
            model_dict.pop(resources_key, None)
        return hash_dict(d)

    @cached_property
    def hash(self) -> str:
        return self.to_hash()

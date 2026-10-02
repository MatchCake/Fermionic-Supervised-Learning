import json
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict

from dataclass_wizard import YAMLWizard
from pythonbasictools.hash_tools import hash_dict

from .dataset_config import DatasetConfig
from .model_config import ModelConfig
from .opt_config import OptConfig


@dataclass
class SingleExperimentConfig(YAMLWizard):
    dataset_config: DatasetConfig
    model: ModelConfig
    opt_config: OptConfig = field(default_factory=OptConfig)

    def __repr__(self):
        return f"SingleExperimentConfig" f"{json.dumps(self.to_dict(), indent=4, default=str)}" f""

    def to_dict(self) -> Dict[str, Any]:
        state = deepcopy(self.__dict__)
        for k, v in state.items():
            if isinstance(v, YAMLWizard):
                state[k] = deepcopy(v.__dict__)
        return state

    def to_hash(self) -> str:
        key_dict = {
            "model": self.model.name,
            **self.model.extra_options,
            **self.dataset_config.__dict__,
        }
        return hash_dict(key_dict)

    @cached_property
    def hash(self) -> str:
        return self.to_hash()

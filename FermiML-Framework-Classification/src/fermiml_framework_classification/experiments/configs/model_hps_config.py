from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List

from dataclass_wizard import YAMLWizard

from .model_config import ModelConfig


@dataclass
class ModelHpsConfig(YAMLWizard):
    name: str
    hyperparameters: Dict[str, List[Any]] = field(default_factory=lambda: defaultdict(list))
    extra_options: Dict[str, Any] = field(default_factory=dict)
    slurm_resources: Dict[str, int] = field(default_factory=lambda: {"mem_mb": 65536, "runtime_min": 960})

    def to_model_configs(self) -> List[ModelConfig]:
        configs = []
        keys = list(sorted(self.hyperparameters.keys()))
        values = [self.hyperparameters[k] for k in keys]
        for hp_values in product(*values):
            configs.append(
                ModelConfig(
                    name=self.name,
                    hyperparameters=dict(zip(keys, hp_values)),
                    extra_options=self.extra_options,
                    slurm_resources=self.slurm_resources,
                )
            )
        return configs

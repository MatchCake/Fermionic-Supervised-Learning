from dataclasses import dataclass, field
from typing import Any, Dict, List

from dataclass_wizard import YAMLWizard


@dataclass
class ModelConfig(YAMLWizard):
    name: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    extra_options: Dict[str, Any] = field(default_factory=dict)
    slurm_resources: Dict[str, int] = field(default_factory=lambda: {"mem_mb": 65536, "runtime_min": 960})

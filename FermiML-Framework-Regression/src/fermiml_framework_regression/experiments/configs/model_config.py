from dataclasses import dataclass, field
from typing import Any, Dict

from dataclass_wizard import YAMLWizard


@dataclass
class ModelConfig(YAMLWizard):
    """Configuration for a single model with fixed hyperparameters.

    :param name: Model name matching :attr:`Model.MODEL_NAME` (case-insensitive).
    :param hyperparameters: Keyword arguments forwarded to the model constructor.
    :param extra_options: Additional keyword arguments forwarded to the model.
    :param slurm_resources: SLURM resource requests (``mem_mb``, ``runtime_min``).
        Excluded from the experiment hash.
    """

    name: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    extra_options: Dict[str, Any] = field(default_factory=dict)
    slurm_resources: Dict[str, int] = field(default_factory=lambda: {"mem_mb": 65536, "runtime_min": 960})

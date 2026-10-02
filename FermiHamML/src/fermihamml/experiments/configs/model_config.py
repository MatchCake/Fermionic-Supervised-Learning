from dataclasses import dataclass, field
from typing import Any, Dict

from dataclass_wizard import YAMLWizard


@dataclass
class ModelConfig(YAMLWizard):
    name: str
    extra_options: Dict[str, Any] = field(default_factory=dict)

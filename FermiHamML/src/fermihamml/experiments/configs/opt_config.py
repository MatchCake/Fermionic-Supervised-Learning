from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from dataclass_wizard import YAMLWizard


@dataclass
class OptConfig(YAMLWizard):
    automl_iterations: int = 32
    inner_max_epochs: int = 128
    inner_max_time: Optional[str] = None
    max_epochs: int = 1024
    max_time: Optional[str] = None
    monitor: str = "val_loss"
    automl_overwrite_fit: bool = False
    accelerator: str = "auto"
    extra_options: Dict[str, Any] = field(default_factory=dict)

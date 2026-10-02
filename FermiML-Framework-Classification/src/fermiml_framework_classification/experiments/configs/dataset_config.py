from dataclasses import dataclass, field
from typing import Any, Dict

from dataclass_wizard import YAMLWizard


@dataclass
class DatasetConfig(YAMLWizard):
    dataset_name: str
    split_id: int = 0
    splits_seed: int = 0
    cv_splits: int = 5
    val_size: float = 0.2
    extra_options: Dict[str, Any] = field(default_factory=dict)

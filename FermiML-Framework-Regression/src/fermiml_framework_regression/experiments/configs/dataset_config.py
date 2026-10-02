from dataclasses import dataclass, field
from typing import Any, Dict

from dataclass_wizard import YAMLWizard


@dataclass
class DatasetConfig(YAMLWizard):
    """Configuration for a single dataset + CV split.

    :param dataset_name: Name matching :attr:`Dataset.DATASET_NAME`.
    :param split_id: CV split index (0-based).
    :param splits_seed: Random seed for the CV splitter.
    :param cv_splits: Total number of CV splits.
    :param val_size: Fraction of the training pool used for validation.
    :param extra_options: Additional keyword arguments forwarded to the dataset.
    """

    dataset_name: str
    split_id: int = 0
    splits_seed: int = 0
    cv_splits: int = 5
    val_size: float = 0.2
    extra_options: Dict[str, Any] = field(default_factory=dict)

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Tuple

from dataclass_wizard import YAMLWizard

from .dataset_config import DatasetConfig


@dataclass
class DatasetsConfigItem(YAMLWizard):
    name: str
    hyperparameters: Dict[str, List[Any]] = field(default_factory=lambda: defaultdict(list))
    extra_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetsConfig(YAMLWizard):
    datasets: List[DatasetsConfigItem]
    splits_seed: int = 0
    cv_splits: int = 5
    val_size: float = 0.2

    def to_dataset_configs(self) -> List[DatasetConfig]:
        dataset_configs = []
        for data_item, split_id in product(self.datasets, range(self.cv_splits)):
            keys = list(sorted(data_item.hyperparameters.keys()))
            values = [data_item.hyperparameters[k] for k in keys]
            for hp_values in product(*values):
                dataset_configs.append(
                    DatasetConfig(
                        dataset_name=data_item.name,
                        split_id=split_id,
                        splits_seed=self.splits_seed,
                        cv_splits=self.cv_splits,
                        val_size=self.val_size,
                        extra_options={
                            **data_item.extra_options,
                            **dict(zip(keys, hp_values)),
                        },
                    )
                )
        return dataset_configs

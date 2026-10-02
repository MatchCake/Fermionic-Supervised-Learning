from dataclasses import dataclass
from typing import List, Tuple

from dataclass_wizard import YAMLWizard

from .dataset_config import DatasetConfig


@dataclass
class DatasetsConfig(YAMLWizard):
    names: List[str]
    num_split_id: int
    batch_sizes: List[int]
    seed: int = 0

    def to_dataset_configs(self) -> List[DatasetConfig]:
        dataset_configs = []
        for name, batch_size in zip(self.names, self.batch_sizes):
            for split_id in range(self.num_split_id):
                dataset_configs.append(
                    DatasetConfig(
                        dataset_name=name,
                        split_id=split_id,
                        batch_size=batch_size,
                        seed=self.seed,
                    )
                )
        return dataset_configs

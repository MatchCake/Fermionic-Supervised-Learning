from dataclasses import dataclass

from dataclass_wizard import YAMLWizard


@dataclass
class DatasetConfig(YAMLWizard):
    dataset_name: str
    split_id: int = 0
    seed: int = 0
    batch_size: int = 256

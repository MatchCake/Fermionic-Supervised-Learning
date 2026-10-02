from pathlib import Path
from typing import Sequence, Union

import torch
from matchcake_opt.datasets import BaseDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Compose


class Cifar2Dataset(BaseDataset):
    """
    https://github.com/UKPLab/qnlpai2025-ham-classifier/blob/40720adc5bfbc6c90f17007fffd904fec4916828/ham_classifier/experiment.py#L222
    """

    DATASET_NAME = "Cifar2"

    @staticmethod
    def to_long_tensor(y: Sequence[int]) -> torch.Tensor:
        return torch.tensor(y, dtype=torch.long)

    @staticmethod
    def binarize_label(label: int) -> int:
        vehicle_labels = [0, 1, 8, 9]  # airplane, automobile, ship, truck
        animal_labels = [2, 3, 4, 5, 6, 7]  # bird, cat, deer, dog, frog, horse
        if label in vehicle_labels:
            return 0  # Vehicule class
        elif label in animal_labels:
            return 1  # Animal class
        else:
            raise ValueError(f"Label {label} is not a valid CIFAR-10 label")

    def __init__(self, data_dir: Union[str, Path] = Path("./data/") / DATASET_NAME, train: bool = True, **kwargs):
        super().__init__(data_dir, train, **kwargs)
        self._data = CIFAR10(
            self.data_dir,
            train=self.train,
            download=True,
            transform=Compose(
                [
                    ToTensor(),
                    transforms.Normalize(0.0, 1.0),
                ]
            ),
            target_transform=Compose([self.binarize_label, self.to_long_tensor]),
        )

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_output_shape(self) -> tuple:
        return (2,)

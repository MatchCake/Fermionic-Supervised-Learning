from pathlib import Path
from typing import Sequence, Union

import torch
from matchcake_opt.datasets import BaseDataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Compose


class FashionDataset(BaseDataset):
    """
    Dataset from: https://arxiv.org/abs/1708.07747
    """

    DATASET_NAME = "Fashion"

    @staticmethod
    def to_long_tensor(y: Sequence[int]) -> torch.Tensor:
        return torch.tensor(y, dtype=torch.long)

    def __init__(self, data_dir: Union[str, Path] = Path("./data/") / DATASET_NAME, train: bool = True, **kwargs):
        super().__init__(data_dir, train, **kwargs)
        self._data = FashionMNIST(
            self.data_dir,
            train=self.train,
            download=True,
            transform=Compose(
                [
                    ToTensor(),
                    transforms.Normalize(0.0, 1.0),
                ]
            ),
            target_transform=Compose([self.to_long_tensor]),
        )

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def get_output_shape(self) -> tuple:
        return (10,)

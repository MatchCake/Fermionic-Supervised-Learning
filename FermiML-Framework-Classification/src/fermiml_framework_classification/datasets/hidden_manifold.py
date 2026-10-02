from typing import Optional, Union

import numpy as np

from .dataset import Dataset, _load_hdf5_split


class HiddenManifold(Dataset):
    DATASET_NAME = "hidden-manifold"
    DEFAULT_DIMENSIONS = 20

    @classmethod
    def feature_dimension(cls, extra_options: Optional[dict] = None) -> int:
        """The ambient dimension, which this dataset sweeps directly through ``dimensions``.

        :param extra_options: the dataset's ``extra_options``; ``dimensions`` defaults to 20.
        :return: the number of input features.
        :rtype: int
        """
        return int((extra_options or {}).get("dimensions", cls.DEFAULT_DIMENSIONS))

    def __init__(
        self,
        *,
        dimensions: Union[int, str] = 20,
        **kwargs,
    ):
        assert 2 <= int(dimensions) <= 20, "Number of dimensions must be between 2 and 20"
        super().__init__(**kwargs)
        self._dimensions = str(dimensions)

    def fetch_x_train(self) -> np.ndarray:
        inputs, _ = _load_hdf5_split(self.DATASET_NAME, "train", self._dimensions)
        return inputs

    def fetch_y_train(self) -> np.ndarray:
        _, labels = _load_hdf5_split(self.DATASET_NAME, "train", self._dimensions)
        return (labels + 1) // 2

    def fetch_x_test(self) -> np.ndarray:
        inputs, _ = _load_hdf5_split(self.DATASET_NAME, "test", self._dimensions)
        return inputs

    def fetch_y_test(self) -> np.ndarray:
        _, labels = _load_hdf5_split(self.DATASET_NAME, "test", self._dimensions)
        return (labels + 1) // 2

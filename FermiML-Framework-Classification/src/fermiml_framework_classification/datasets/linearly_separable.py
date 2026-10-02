from typing import Optional, Union

import numpy as np

from .dataset import Dataset, _load_hdf5_split


class LinearlySeparable(Dataset):
    DATASET_NAME = "linearly-separable"
    DEFAULT_DIMENSION = 20

    @classmethod
    def feature_dimension(cls, extra_options: Optional[dict] = None) -> int:
        """The ambient dimension, which this dataset sweeps directly through ``dimension``.

        :param extra_options: the dataset's ``extra_options``; ``dimension`` defaults to 20.
        :return: the number of input features.
        :rtype: int
        """
        return int((extra_options or {}).get("dimension", cls.DEFAULT_DIMENSION))

    def __init__(
        self,
        *,
        dimension: Union[int, str] = 20,
        **kwargs,
    ):
        assert 2 <= int(dimension) <= 20, "Number of manifolds must be between 2 and 20"
        super().__init__(**kwargs)
        self._dimension = str(dimension)

    def fetch_x_train(self) -> np.ndarray:
        inputs, _ = _load_hdf5_split(self.DATASET_NAME, "train", self._dimension)
        return inputs

    def fetch_y_train(self) -> np.ndarray:
        _, labels = _load_hdf5_split(self.DATASET_NAME, "train", self._dimension)
        return (labels + 1) // 2

    def fetch_x_test(self) -> np.ndarray:
        inputs, _ = _load_hdf5_split(self.DATASET_NAME, "test", self._dimension)
        return inputs

    def fetch_y_test(self) -> np.ndarray:
        _, labels = _load_hdf5_split(self.DATASET_NAME, "test", self._dimension)
        return (labels + 1) // 2

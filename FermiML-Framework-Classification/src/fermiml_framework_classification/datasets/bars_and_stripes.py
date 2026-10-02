from typing import Optional, Union

import numpy as np

from .dataset import Dataset, _load_hdf5_split


class BarsAndStripes(Dataset):
    DATASET_NAME = "bars-and-stripes"
    DEFAULT_IMAGE_SIZE = 4

    @classmethod
    def feature_dimension(cls, extra_options: Optional[dict] = None) -> int:
        """The flattened image size, i.e. ``image_size ** 2`` (an image size of 4 gives 16 features).

        :param extra_options: the dataset's ``extra_options``; ``image_size`` defaults to 4.
        :return: the number of input features.
        :rtype: int
        """
        image_size = int((extra_options or {}).get("image_size", cls.DEFAULT_IMAGE_SIZE))
        return image_size**2

    def __init__(
        self,
        *,
        image_size: Union[int, str] = 4,
        **kwargs,
    ):
        assert int(image_size) in [4, 8, 16, 32], "Image size must be one of 4, 8, 16, 32"
        super().__init__(**kwargs)
        self._image_size = str(image_size)

    def fetch_x_train(self) -> np.ndarray:
        inputs, _ = _load_hdf5_split(self.DATASET_NAME, "train", self._image_size)
        return inputs

    def fetch_y_train(self) -> np.ndarray:
        _, labels = _load_hdf5_split(self.DATASET_NAME, "train", self._image_size)
        return (labels + 1) // 2

    def fetch_x_test(self) -> np.ndarray:
        inputs, _ = _load_hdf5_split(self.DATASET_NAME, "test", self._image_size)
        return inputs

    def fetch_y_test(self) -> np.ndarray:
        _, labels = _load_hdf5_split(self.DATASET_NAME, "test", self._image_size)
        return (labels + 1) // 2

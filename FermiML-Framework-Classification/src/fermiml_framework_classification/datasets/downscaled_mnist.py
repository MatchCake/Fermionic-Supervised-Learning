from functools import cached_property
from typing import Optional, Union

import numpy as np

from .dataset import Dataset, _datasets_root


class DownscaledMnist(Dataset):
    DATASET_NAME = "downscaled-mnist"
    NUMBER_OF_TRAINING_SAMPLES = 11552
    NUMBER_OF_TEST_SAMPLES = 1902
    DEFAULT_DIMENSION = 20

    @classmethod
    def feature_dimension(cls, extra_options: Optional[dict] = None) -> int:
        """The number of retained principal components, swept directly through ``dimension``.

        :param extra_options: the dataset's ``extra_options``; ``dimension`` defaults to 20.
        :return: the number of input features.
        :rtype: int
        """
        return int((extra_options or {}).get("dimension", cls.DEFAULT_DIMENSION))

    def __init__(
        self,
        *,
        dimension: Union[int, str] = 20,
        n_train_samples: int = 250,
        n_test_samples: int = 250,
        **kwargs,
    ):
        assert 2 <= int(dimension) <= 20, "Number of dimensions must be between 2 and 20"
        assert (
            0 < n_train_samples <= self.NUMBER_OF_TRAINING_SAMPLES
        ), "Number of training samples must be between 0 and 11552"
        assert 0 < n_test_samples <= self.NUMBER_OF_TEST_SAMPLES, "Number of test samples must be between 0 and 1902"
        super().__init__(**kwargs)
        self._dimension = str(dimension)
        self._rng = np.random.default_rng(self._cv_split_id)
        self._train_indexes = self._rng.choice(self.NUMBER_OF_TRAINING_SAMPLES, n_train_samples, replace=False)
        self._test_indexes = self._rng.choice(self.NUMBER_OF_TEST_SAMPLES, n_test_samples, replace=False)

    @cached_property
    def _pennylane_dataset(self):
        # Load once into the bind-mounted datasets directory so the cluster
        # compute nodes never need to reach PennyLane cloud.
        import pennylane as qp

        folder = _datasets_root()
        folder.mkdir(parents=True, exist_ok=True)
        [ds] = qp.data.load("other", name="downscaled-mnist", folder_path=folder)
        return ds

    def fetch_x_train(self) -> np.ndarray:
        x_train = np.asarray(self._pennylane_dataset.train[self._dimension]["inputs"])
        x_train = x_train[self._train_indexes]
        return x_train

    def fetch_y_train(self) -> np.ndarray:
        y_train = np.asarray(self._pennylane_dataset.train[self._dimension]["labels"])
        y_train = y_train[self._train_indexes]
        # make labels 0, 1
        y_train = (y_train + 1) // 2
        return y_train

    def fetch_x_test(self) -> np.ndarray:
        x_test = np.asarray(self._pennylane_dataset.test[self._dimension]["inputs"])
        x_test = x_test[self._test_indexes]
        return x_test

    def fetch_y_test(self) -> np.ndarray:
        y_test = np.asarray(self._pennylane_dataset.test[self._dimension]["labels"])
        y_test = y_test[self._test_indexes]
        # make labels 0, 1
        y_test = (y_test + 1) // 2
        return y_test

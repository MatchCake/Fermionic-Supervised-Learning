from functools import cached_property
from typing import Tuple

import numpy as np

from .dataset import Dataset


class CaliforniaHousingDataset(Dataset):
    """California Housing regression dataset.

    Predicts median house values from 8 geographic and demographic features
    (total 20,640 samples in the original). Subsampled to make kernel methods
    tractable: ``max_train_samples`` are used as the training pool and
    ``max_test_samples`` as the held-out test set. The full dataset is shuffled
    once with ``random_state`` before splitting, so the split is deterministic
    but different from the canonical sklearn ordering.

    :param max_train_samples: Maximum number of samples in the training pool
        (train + val). Capped to 80% of the available data.
    :param max_test_samples: Maximum number of held-out test samples.
    :param random_state: Seed used to shuffle the dataset before splitting.
    """

    DATASET_NAME = "california-housing"
    DEFAULT_MAX_TRAIN_SAMPLES = 500
    DEFAULT_MAX_TEST_SAMPLES = 100
    DEFAULT_RANDOM_STATE = 0

    def __init__(
        self,
        *,
        max_train_samples: int = DEFAULT_MAX_TRAIN_SAMPLES,
        max_test_samples: int = DEFAULT_MAX_TEST_SAMPLES,
        random_state: int = DEFAULT_RANDOM_STATE,
        cv_split_id: int = Dataset.DEFAULT_CV_SPLIT_ID,
        cv_splits: int = Dataset.DEFAULT_CV_SPLITS,
        cv_random_state: int = Dataset.DEFAULT_CV_RANDOM_STATE,
        val_size: float = Dataset.DEFAULT_VAL_SIZE,
    ):
        super().__init__(
            cv_split_id=cv_split_id,
            cv_splits=cv_splits,
            cv_random_state=cv_random_state,
            val_size=val_size,
        )
        self.max_train_samples = max_train_samples
        self.max_test_samples = max_test_samples
        self.random_state = random_state

    @cached_property
    def _split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing()
        x_all = data.data.astype(np.float32)
        y_all = data.target.astype(np.float32)

        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(len(x_all))
        x_all, y_all = x_all[idx], y_all[idx]

        n_train = min(self.max_train_samples, int(0.8 * len(x_all)))
        n_test = min(self.max_test_samples, len(x_all) - n_train)
        return x_all[:n_train], y_all[:n_train], x_all[n_train : n_train + n_test], y_all[n_train : n_train + n_test]

    def fetch_x_train(self) -> np.ndarray:
        return self._split_data[0]

    def fetch_y_train(self) -> np.ndarray:
        return self._split_data[1]

    def fetch_x_test(self) -> np.ndarray:
        return self._split_data[2]

    def fetch_y_test(self) -> np.ndarray:
        return self._split_data[3]

from functools import cached_property
from typing import Tuple

import numpy as np
from sklearn.datasets import make_friedman1

from .dataset import Dataset


class FriedmanDataset(Dataset):
    """Friedman #1 synthetic regression dataset.

    Targets follow :math:`y = 10\\sin(\\pi x_1 x_2) + 20(x_3 - 0.5)^2 + 10 x_4 + 5 x_5 + \\varepsilon`
    with 10 input features, only 5 of which are informative. Generated via
    :func:`sklearn.datasets.make_friedman1`.

    :param n_samples: Number of training-pool samples.
    :param n_test_samples: Number of held-out test samples. Generated with a
        different random seed (``random_state + 1000``) so the test set is
        independent of the training pool.
    :param noise: Standard deviation of the Gaussian noise added to targets.
        Set to 0 for the test set so it reflects the clean ground truth.
    :param random_state: Base random seed for reproducibility.
    """

    DATASET_NAME = "friedman"
    DEFAULT_N_SAMPLES = 500
    DEFAULT_N_TEST_SAMPLES = 100
    DEFAULT_NOISE = 0.1
    DEFAULT_RANDOM_STATE = 0

    def __init__(
        self,
        *,
        n_samples: int = DEFAULT_N_SAMPLES,
        n_test_samples: int = DEFAULT_N_TEST_SAMPLES,
        noise: float = DEFAULT_NOISE,
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
        self.n_samples = n_samples
        self.n_test_samples = n_test_samples
        self.noise = noise
        self.random_state = random_state

    @cached_property
    def _train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y = make_friedman1(n_samples=self.n_samples, noise=self.noise, random_state=self.random_state)
        return x.astype(np.float32), y.astype(np.float32)

    @cached_property
    def _test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y = make_friedman1(n_samples=self.n_test_samples, noise=0.0, random_state=self.random_state + 1000)
        return x.astype(np.float32), y.astype(np.float32)

    def fetch_x_train(self) -> np.ndarray:
        return self._train_data[0]

    def fetch_y_train(self) -> np.ndarray:
        return self._train_data[1]

    def fetch_x_test(self) -> np.ndarray:
        return self._test_data[0]

    def fetch_y_test(self) -> np.ndarray:
        return self._test_data[1]

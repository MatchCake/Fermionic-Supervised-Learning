from functools import cached_property
from typing import Optional

import numpy as np
from sklearn.model_selection import ShuffleSplit


class Dataset:
    """Base class for regression datasets.

    Subclasses must implement :meth:`fetch_x_train`, :meth:`fetch_y_train`,
    :meth:`fetch_x_test`, and :meth:`fetch_y_test`. After :meth:`prepare_data`
    is called, the split attributes ``x_train``, ``y_train``, ``x_val``,
    ``y_val``, ``x_test``, and ``y_test`` become available.

    :param cv_split_id: Index of the CV split to use (0-based).
    :param cv_splits: Total number of CV splits.
    :param cv_random_state: Random seed for the CV splitter.
    :param val_size: Fraction of the training pool reserved for validation.
    """

    DATASET_NAME = "Dataset"
    DEFAULT_VAL_SIZE = 0.2
    DEFAULT_CV_SPLITS = 20
    DEFAULT_CV_SPLIT_ID = 0
    DEFAULT_CV_RANDOM_STATE = 0

    def __init__(
        self,
        *,
        cv_split_id: int = DEFAULT_CV_SPLIT_ID,
        cv_splits: int = DEFAULT_CV_SPLITS,
        cv_random_state: int = DEFAULT_CV_RANDOM_STATE,
        val_size: float = DEFAULT_VAL_SIZE,
    ):
        assert 0 <= val_size <= 1, "Validation size must be between 0 and 1"
        assert cv_splits > 0, "Number of CV splits must be greater than 0"
        assert 0 <= cv_split_id < cv_splits, "CV split id must be between 0 and number of CV splits"
        self._cv_split_id = cv_split_id
        self._cv = ShuffleSplit(n_splits=cv_splits, test_size=val_size, random_state=cv_random_state)
        self._is_prepared_flag = False
        self._raw_x_train: Optional[np.ndarray] = None
        self._x_train: Optional[np.ndarray] = None
        self._raw_y_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._x_val: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._x_test: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None

    def fetch_x_train(self) -> np.ndarray:
        raise NotImplementedError()

    def fetch_y_train(self) -> np.ndarray:
        raise NotImplementedError()

    def fetch_x_test(self) -> np.ndarray:
        raise NotImplementedError()

    def fetch_y_test(self) -> np.ndarray:
        raise NotImplementedError()

    def prepare_data(self) -> "Dataset":
        if self._is_prepared_flag:
            return self
        self._raw_x_train = self.fetch_x_train()
        self._raw_y_train = self.fetch_y_train()
        self._x_test = self.fetch_x_test()
        self._y_test = self.fetch_y_test()

        [(train_idx, val_idx)] = [
            (ti, vi) for i, (ti, vi) in enumerate(self._cv.split(self._raw_x_train)) if i == self._cv_split_id
        ]
        self._x_train = self._raw_x_train[train_idx]
        self._y_train = self._raw_y_train[train_idx]
        self._x_val = self._raw_x_train[val_idx]
        self._y_val = self._raw_y_train[val_idx]
        self._is_prepared_flag = True
        return self

    @cached_property
    def x_train(self) -> np.ndarray:
        self.prepare_data()
        return self._x_train  # type: ignore[return-value]

    @cached_property
    def y_train(self) -> np.ndarray:
        self.prepare_data()
        return self._y_train  # type: ignore[return-value]

    @cached_property
    def x_val(self) -> np.ndarray:
        self.prepare_data()
        return self._x_val  # type: ignore[return-value]

    @cached_property
    def y_val(self) -> np.ndarray:
        self.prepare_data()
        return self._y_val  # type: ignore[return-value]

    @cached_property
    def x_test(self) -> np.ndarray:
        self.prepare_data()
        return self._x_test  # type: ignore[return-value]

    @cached_property
    def y_test(self) -> np.ndarray:
        self.prepare_data()
        return self._y_test  # type: ignore[return-value]

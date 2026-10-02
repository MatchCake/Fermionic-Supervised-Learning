import os
from functools import cached_property
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def _datasets_root() -> Path:
    """Return the local datasets directory, derived from DATA_DIR env var (defaults to 'data')."""
    return Path(os.environ.get("DATA_DIR", "data")).parent / "datasets"


def _download_dataset(dataset_name: str) -> None:
    folder = _datasets_root()
    folder.mkdir(parents=True, exist_ok=True)
    try:
        import pennylane as qp

        qp.data.load("other", name=dataset_name, folder_path=folder)
    except Exception as e:
        raise RuntimeError(
            f"Dataset '{dataset_name}' not found at {folder / dataset_name} and automatic download failed: {e}. "
            f"Please download all datasets manually."
        ) from e


def _load_hdf5_split(dataset_name: str, split: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Load inputs and labels from a locally cached PennyLane HDF5 dataset file."""
    hdf5_path = _datasets_root() / dataset_name / f"{dataset_name}.h5"
    if not hdf5_path.exists():
        _download_dataset(dataset_name)
    with h5py.File(hdf5_path, "r") as f:
        grp = f[split][key]
        n = len(grp["inputs"])
        n_features = len(grp["inputs"]["0"])
        inputs = np.array([[grp["inputs"][str(i)][str(j)][()] for j in range(n_features)] for i in range(n)])
        labels = np.array([grp["labels"][str(i)][()] for i in range(len(grp["labels"]))])
    return inputs, labels


class Dataset:
    DATASET_NAME = "Dataset"
    DEFAULT_VAL_SIZE = 0.2
    DEFAULT_CV_SPLITS = 20
    DEFAULT_CV_SPLIT_ID = 0
    DEFAULT_CV_RANDOM_STATE = 0

    @classmethod
    def feature_dimension(cls, extra_options: Optional[dict] = None) -> int:
        """Return the input dimension this dataset produces for the given extra options.

        Known before any data is loaded, so the experiment grid can drop model configurations that the
        dimension makes infeasible (see ``ExperimentConfig._model_config_is_feasible``) instead of
        submitting jobs that are certain to fail.

        :param extra_options: the dataset's ``extra_options``; defaults are used for missing keys.
        :return: the number of input features.
        :rtype: int
        """
        raise NotImplementedError()

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
        self._cv = StratifiedShuffleSplit(n_splits=cv_splits, test_size=val_size, random_state=cv_random_state)
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

    def fetch_(self):
        self._raw_x_train = self.fetch_x_train()
        self._raw_y_train = self.fetch_y_train()
        self._x_test = self.fetch_x_test()
        self._y_test = self.fetch_y_test()
        return self

    def prepare_data(self):
        if self._is_prepared_flag:
            return self
        self._raw_x_train = self.fetch_x_train()
        self._raw_y_train = self.fetch_y_train()
        self._x_test = self.fetch_x_test()
        self._y_test = self.fetch_y_test()

        [(train_idx, val_idx)] = [
            (ti, vi)
            for i, (ti, vi) in enumerate(self._cv.split(self._raw_x_train, self._raw_y_train))
            if i == self._cv_split_id
        ]
        self._x_train = self._raw_x_train[train_idx]
        self._y_train = self._raw_y_train[train_idx]
        self._x_val = self._raw_x_train[val_idx]
        self._y_val = self._raw_y_train[val_idx]
        self._is_prepared_flag = True
        return self

    @cached_property
    def x_train(self):
        self.prepare_data()
        return self._x_train

    @cached_property
    def y_train(self):
        self.prepare_data()
        return self._y_train

    @cached_property
    def x_val(self):
        self.prepare_data()
        return self._x_val

    @cached_property
    def y_val(self):
        self.prepare_data()
        return self._y_val

    @cached_property
    def x_test(self):
        self.prepare_data()
        return self._x_test

    @cached_property
    def y_test(self):
        self.prepare_data()
        return self._y_test

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fermiml_framework_classification.datasets.dataset import (
    Dataset,
    _download_dataset,
    _load_hdf5_split,
)


class ConcreteDataset(Dataset):
    def __init__(self, n_samples=100, n_features=4, n_classes=2, **kwargs):
        super().__init__(**kwargs)
        rng = np.random.default_rng(0)
        self._X = rng.random((n_samples, n_features)).astype(np.float32)
        self._y = (np.arange(n_samples) % n_classes).astype(int)
        self._X_test = rng.random((20, n_features)).astype(np.float32)
        self._y_test = (np.arange(20) % n_classes).astype(int)

    def fetch_x_train(self):
        return self._X

    def fetch_y_train(self):
        return self._y

    def fetch_x_test(self):
        return self._X_test

    def fetch_y_test(self):
        return self._y_test


class TestDataset:
    @pytest.fixture
    def dataset(self):
        return ConcreteDataset()

    def test_dataset_name(self):
        assert Dataset.DATASET_NAME == "Dataset"

    def test_default_init(self, dataset):
        assert dataset._cv_split_id == Dataset.DEFAULT_CV_SPLIT_ID
        assert dataset._is_prepared_flag is False

    def test_custom_init(self):
        ds = ConcreteDataset(cv_split_id=1, cv_splits=5, cv_random_state=42, val_size=0.1)
        assert ds._cv_split_id == 1
        assert ds._cv.n_splits == 5
        assert ds._cv.test_size == 0.1

    def test_invalid_val_size(self):
        with pytest.raises(AssertionError):
            ConcreteDataset(val_size=1.5)

    def test_invalid_cv_splits(self):
        with pytest.raises(AssertionError):
            ConcreteDataset(cv_splits=0)

    def test_invalid_cv_split_id(self):
        with pytest.raises(AssertionError):
            ConcreteDataset(cv_splits=5, cv_split_id=5)

    def test_fetch_not_implemented(self):
        ds = Dataset.__new__(Dataset)
        ds.__init__(cv_splits=1, cv_split_id=0)
        with pytest.raises(NotImplementedError):
            ds.fetch_x_train()
        with pytest.raises(NotImplementedError):
            ds.fetch_y_train()
        with pytest.raises(NotImplementedError):
            ds.fetch_x_test()
        with pytest.raises(NotImplementedError):
            ds.fetch_y_test()

    def test_prepare_data_sets_flag(self, dataset):
        assert dataset._is_prepared_flag is False
        dataset.prepare_data()
        assert dataset._is_prepared_flag is True

    def test_prepare_data_idempotent(self, dataset):
        dataset.prepare_data()
        x_train_first = dataset._x_train
        dataset.prepare_data()
        assert dataset._x_train is x_train_first

    def test_prepare_data_splits(self, dataset):
        dataset.prepare_data()
        n_raw = len(dataset._raw_x_train)
        n_train = len(dataset._x_train)
        n_val = len(dataset._x_val)
        assert n_train + n_val == n_raw

    def test_x_train_shape(self, dataset):
        assert dataset.x_train.shape[0] > 0
        assert dataset.x_train.shape[1] == 4

    def test_y_train_length(self, dataset):
        assert len(dataset.y_train) == len(dataset.x_train)

    def test_x_val_shape(self, dataset):
        assert dataset.x_val.shape[0] > 0
        assert dataset.x_val.shape[1] == 4

    def test_y_val_length(self, dataset):
        assert len(dataset.y_val) == len(dataset.x_val)

    def test_x_test_shape(self, dataset):
        assert dataset.x_test.shape == (20, 4)

    def test_y_test_length(self, dataset):
        assert len(dataset.y_test) == 20

    def test_train_val_no_overlap(self, dataset):
        dataset.prepare_data()
        train_set = set(map(tuple, dataset._x_train))
        val_set = set(map(tuple, dataset._x_val))
        assert train_set.isdisjoint(val_set)

    def test_different_cv_splits_give_different_data(self):
        ds0 = ConcreteDataset(cv_split_id=0, cv_splits=5)
        ds1 = ConcreteDataset(cv_split_id=1, cv_splits=5)
        assert not np.array_equal(ds0.x_train, ds1.x_train)

    def test_val_size_respected(self):
        ds = ConcreteDataset(val_size=0.2, cv_splits=1)
        ds.prepare_data()
        n_raw = len(ds._raw_x_train)
        expected_val = round(n_raw * 0.2)
        assert abs(len(ds._x_val) - expected_val) <= 2

    def test_fetch_underscore_populates_raw_fields(self, dataset):
        dataset.fetch_()
        assert dataset._raw_x_train is not None
        assert dataset._raw_y_train is not None
        assert dataset._x_test is not None
        assert dataset._y_test is not None

    def test_fetch_underscore_returns_self(self, dataset):
        assert dataset.fetch_() is dataset


class TestDownloadDataset:
    _PATCH_ROOT = "fermiml_framework_classification.datasets.dataset._datasets_root"

    def test_download_creates_folder_and_calls_pennylane(self, tmp_path):
        mock_qp = MagicMock()
        with patch(self._PATCH_ROOT, return_value=tmp_path), patch.dict(sys.modules, {"pennylane": mock_qp}):
            _download_dataset("some-dataset")
        mock_qp.data.load.assert_called_once_with("other", name="some-dataset", folder_path=tmp_path)

    def test_download_raises_runtime_error_on_failure(self, tmp_path):
        mock_qp = MagicMock()
        mock_qp.data.load.side_effect = ImportError("aiohttp broken")
        with patch(self._PATCH_ROOT, return_value=tmp_path), patch.dict(sys.modules, {"pennylane": mock_qp}):
            with pytest.raises(RuntimeError, match="automatic download failed"):
                _download_dataset("some-dataset")


class TestLoadHdf5Split:
    _PATCH_ROOT = "fermiml_framework_classification.datasets.dataset._datasets_root"

    def test_triggers_download_when_file_missing(self, tmp_path):
        with (
            patch(self._PATCH_ROOT, return_value=tmp_path),
            patch("fermiml_framework_classification.datasets.dataset._download_dataset") as mock_dl,
        ):
            mock_dl.side_effect = RuntimeError("no network")
            with pytest.raises(RuntimeError):
                _load_hdf5_split("some-dataset", "train", "5")
        mock_dl.assert_called_once_with("some-dataset")

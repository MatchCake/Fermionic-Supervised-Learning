from unittest.mock import patch

import numpy as np
import pytest

from fermiml_framework_classification.datasets.linearly_separable import (
    LinearlySeparable,
)

_PATCH = "fermiml_framework_classification.datasets.linearly_separable._load_hdf5_split"


class TestLinearlySeparable:
    @pytest.fixture
    def dataset(self):
        return LinearlySeparable(dimension=10, cv_splits=1)

    # --- init ---

    @pytest.mark.parametrize("d", [2, 5, 10, 20])
    def test_valid_dimension(self, d):
        ds = LinearlySeparable(dimension=d, cv_splits=1)
        assert ds._dimension == str(d)

    def test_dimension_as_string(self):
        ds = LinearlySeparable(dimension="10", cv_splits=1)
        assert ds._dimension == "10"

    def test_invalid_dimension_too_low(self):
        with pytest.raises(AssertionError):
            LinearlySeparable(dimension=1)

    def test_invalid_dimension_too_high(self):
        with pytest.raises(AssertionError):
            LinearlySeparable(dimension=21)

    def test_inherits_dataset(self):
        from fermiml_framework_classification.datasets.dataset import Dataset

        assert issubclass(LinearlySeparable, Dataset)

    # --- fetch methods ---

    def test_fetch_x_train_returns_array(self, dataset):
        assert isinstance(dataset.fetch_x_train(), np.ndarray)

    def test_fetch_y_train_returns_array(self, dataset):
        assert isinstance(dataset.fetch_y_train(), np.ndarray)

    def test_fetch_x_test_returns_array(self, dataset):
        assert isinstance(dataset.fetch_x_test(), np.ndarray)

    def test_fetch_y_test_returns_array(self, dataset):
        assert isinstance(dataset.fetch_y_test(), np.ndarray)

    def test_fetch_x_train_shape(self, dataset):
        result = dataset.fetch_x_train()
        assert result.ndim == 2
        assert result.shape[1] == 10

    def test_fetch_x_test_shape(self, dataset):
        result = dataset.fetch_x_test()
        assert result.ndim == 2
        assert result.shape[1] == 10

    def test_fetch_y_train_labels_are_binary(self, dataset):
        assert set(dataset.fetch_y_train()).issubset({0, 1})

    def test_fetch_y_test_labels_are_binary(self, dataset):
        assert set(dataset.fetch_y_test()).issubset({0, 1})

    def test_fetch_calls_load_with_dataset_name(self):
        with patch(_PATCH) as m:
            m.return_value = (np.zeros((4, 10)), np.ones(4))
            LinearlySeparable(dimension=10, cv_splits=1).fetch_x_train()
        assert m.call_args[0][0] == "linearly-separable"

    def test_fetch_uses_dimension_as_key(self):
        ds = LinearlySeparable(dimension=5, cv_splits=1)
        result = ds.fetch_x_train()
        assert result.shape[1] == 5

    # --- prepare_data / properties ---

    def test_x_train_property(self, dataset):
        x = dataset.x_train
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2

    def test_y_train_length_matches_x_train(self, dataset):
        assert len(dataset.y_train) == len(dataset.x_train)

    def test_x_val_property(self, dataset):
        x = dataset.x_val
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2

    def test_y_val_length_matches_x_val(self, dataset):
        assert len(dataset.y_val) == len(dataset.x_val)

    def test_x_test_property(self, dataset):
        assert isinstance(dataset.x_test, np.ndarray)

    def test_train_val_sizes_sum_to_raw(self, dataset):
        dataset.prepare_data()
        assert len(dataset._x_train) + len(dataset._x_val) == len(dataset._raw_x_train)

    def test_feature_dimension_follows_dimension(self):
        assert LinearlySeparable.feature_dimension({"dimension": 15}) == 15

    def test_feature_dimension_defaults(self):
        assert LinearlySeparable.feature_dimension() == LinearlySeparable.DEFAULT_DIMENSION

from unittest.mock import patch

import numpy as np
import pytest

from fermiml_framework_classification.datasets.hidden_manifold import HiddenManifold

_PATCH = "fermiml_framework_classification.datasets.hidden_manifold._load_hdf5_split"


class TestHiddenManifold:
    @pytest.fixture
    def dataset(self):
        return HiddenManifold(dimensions=4, cv_splits=1)

    # --- init ---

    @pytest.mark.parametrize("d", [2, 4, 10, 20])
    def test_valid_dimensions(self, d):
        ds = HiddenManifold(dimensions=d, cv_splits=1)
        assert ds._dimensions == str(d)

    def test_dimensions_as_string(self):
        ds = HiddenManifold(dimensions="4", cv_splits=1)
        assert ds._dimensions == "4"

    def test_invalid_dimensions_too_low(self):
        with pytest.raises(AssertionError):
            HiddenManifold(dimensions=1)

    def test_invalid_dimensions_too_high(self):
        with pytest.raises(AssertionError):
            HiddenManifold(dimensions=21)

    def test_inherits_dataset(self):
        from fermiml_framework_classification.datasets.dataset import Dataset

        assert issubclass(HiddenManifold, Dataset)

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

    def test_fetch_x_test_shape(self, dataset):
        result = dataset.fetch_x_test()
        assert result.ndim == 2

    def test_fetch_y_train_labels_are_binary(self, dataset):
        assert set(dataset.fetch_y_train()).issubset({0, 1})

    def test_fetch_y_test_labels_are_binary(self, dataset):
        assert set(dataset.fetch_y_test()).issubset({0, 1})

    def test_fetch_calls_load_with_dataset_name(self):
        with patch(_PATCH) as m:
            m.return_value = (np.zeros((4, 4)), np.ones(4))
            HiddenManifold(dimensions=4, cv_splits=1).fetch_x_train()
        assert m.call_args[0][0] == "hidden-manifold"

    def test_fetch_uses_dimensions_as_key(self):
        ds = HiddenManifold(dimensions=6, cv_splits=1)
        result = ds.fetch_x_train()
        assert result.shape[1] == 6

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

    def test_feature_dimension_follows_dimensions(self):
        assert HiddenManifold.feature_dimension({"dimensions": 6}) == 6

    def test_feature_dimension_defaults(self):
        assert HiddenManifold.feature_dimension() == HiddenManifold.DEFAULT_DIMENSIONS

from unittest.mock import patch

import numpy as np
import pytest

from fermiml_framework_classification.datasets.bars_and_stripes import BarsAndStripes

_PATCH = "fermiml_framework_classification.datasets.bars_and_stripes._load_hdf5_split"


class TestBarsAndStripes:
    @pytest.fixture
    def dataset(self):
        return BarsAndStripes(image_size=4, cv_splits=1)

    # --- init ---

    @pytest.mark.parametrize("size", [4, 8, 16, 32])
    def test_valid_image_sizes(self, size):
        ds = BarsAndStripes(image_size=size, cv_splits=1)
        assert ds._image_size == str(size)

    def test_image_size_as_string(self):
        ds = BarsAndStripes(image_size="4", cv_splits=1)
        assert ds._image_size == "4"

    def test_invalid_image_size(self):
        with pytest.raises(AssertionError):
            BarsAndStripes(image_size=5)

    def test_invalid_image_size_zero(self):
        with pytest.raises(AssertionError):
            BarsAndStripes(image_size=0)

    def test_inherits_dataset(self):
        from fermiml_framework_classification.datasets.dataset import Dataset

        assert issubclass(BarsAndStripes, Dataset)

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
        assert result.shape[1] == 4**2

    def test_fetch_x_test_shape(self, dataset):
        result = dataset.fetch_x_test()
        assert result.ndim == 2
        assert result.shape[1] == 4**2

    def test_fetch_y_train_labels_are_binary(self, dataset):
        assert set(dataset.fetch_y_train()).issubset({0, 1})

    def test_fetch_y_test_labels_are_binary(self, dataset):
        assert set(dataset.fetch_y_test()).issubset({0, 1})

    def test_fetch_y_train_converts_minus_one_to_zero(self):
        with patch(_PATCH) as m:
            m.return_value = (np.zeros((4, 16)), np.array([-1.0, 1.0, -1.0, 1.0]))
            result = BarsAndStripes(image_size=4, cv_splits=1).fetch_y_train()
        np.testing.assert_array_equal(result, [0, 1, 0, 1])

    def test_fetch_y_test_converts_minus_one_to_zero(self):
        with patch(_PATCH) as m:
            m.return_value = (np.zeros((4, 16)), np.array([-1.0, 1.0, -1.0, 1.0]))
            result = BarsAndStripes(image_size=4, cv_splits=1).fetch_y_test()
        np.testing.assert_array_equal(result, [0, 1, 0, 1])

    def test_fetch_calls_load_with_dataset_name(self):
        with patch(_PATCH) as m:
            m.return_value = (np.zeros((4, 16)), np.ones(4))
            BarsAndStripes(image_size=4, cv_splits=1).fetch_x_train()
        assert m.call_args[0][0] == "bars-and-stripes"

    def test_fetch_uses_image_size_as_key(self):
        ds = BarsAndStripes(image_size=8, cv_splits=1)
        result = ds.fetch_x_train()
        assert result.shape[1] == 8**2

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

    def test_feature_dimension_squares_the_image_size(self):
        assert BarsAndStripes.feature_dimension({"image_size": 4}) == 16
        assert BarsAndStripes.feature_dimension({"image_size": 8}) == 64

    def test_feature_dimension_accepts_a_string(self):
        assert BarsAndStripes.feature_dimension({"image_size": "16"}) == 256

    def test_feature_dimension_defaults(self):
        assert BarsAndStripes.feature_dimension() == BarsAndStripes.DEFAULT_IMAGE_SIZE**2

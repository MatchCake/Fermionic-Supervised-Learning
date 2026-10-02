from unittest.mock import patch

import numpy as np
import pytest

from fermiml_framework_classification.datasets.hidden_manifold_diff import (
    HiddenManifoldDiff,
)

_PATCH = "fermiml_framework_classification.datasets.hidden_manifold_diff._load_hdf5_split"


class TestHiddenManifoldDiff:
    @pytest.fixture
    def dataset(self):
        return HiddenManifoldDiff(n_manifolds=4, cv_splits=1)

    # --- init ---

    @pytest.mark.parametrize("n", [2, 4, 10, 20])
    def test_valid_n_manifolds(self, n):
        ds = HiddenManifoldDiff(n_manifolds=n, cv_splits=1)
        assert ds._n_manifolds == str(n)

    def test_n_manifolds_as_string(self):
        ds = HiddenManifoldDiff(n_manifolds="4", cv_splits=1)
        assert ds._n_manifolds == "4"

    def test_invalid_n_manifolds_too_low(self):
        with pytest.raises(AssertionError):
            HiddenManifoldDiff(n_manifolds=1)

    def test_invalid_n_manifolds_too_high(self):
        with pytest.raises(AssertionError):
            HiddenManifoldDiff(n_manifolds=21)

    def test_inherits_dataset(self):
        from fermiml_framework_classification.datasets.dataset import Dataset

        assert issubclass(HiddenManifoldDiff, Dataset)

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
        assert dataset.fetch_x_train().ndim == 2

    def test_fetch_x_test_shape(self, dataset):
        assert dataset.fetch_x_test().ndim == 2

    def test_fetch_calls_load_with_dataset_name(self):
        with patch(_PATCH) as m:
            m.return_value = (np.zeros((4, 4)), np.ones(4))
            HiddenManifoldDiff(n_manifolds=4, cv_splits=1).fetch_x_train()
        assert m.call_args[0][0] == "hidden-manifold"

    def test_fetch_x_train_uses_diff_train_split(self):
        with patch(_PATCH) as m:
            m.return_value = (np.zeros((4, 4)), np.ones(4))
            HiddenManifoldDiff(n_manifolds=4, cv_splits=1).fetch_x_train()
        assert m.call_args[0][1] == "diff_train"

    def test_fetch_x_test_uses_diff_test_split(self):
        with patch(_PATCH) as m:
            m.return_value = (np.zeros((4, 4)), np.ones(4))
            HiddenManifoldDiff(n_manifolds=4, cv_splits=1).fetch_x_test()
        assert m.call_args[0][1] == "diff_test"

    def test_fetch_uses_n_manifolds_as_key(self):
        with patch(_PATCH) as m:
            m.return_value = (np.zeros((4, 10)), np.ones(4))
            HiddenManifoldDiff(n_manifolds=6, cv_splits=1).fetch_x_train()
        assert m.call_args[0][2] == "6"

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

    def test_feature_dimension_is_constant(self):
        # n_manifolds selects the number of latent manifolds, not the ambient dimension.
        assert HiddenManifoldDiff.feature_dimension({"n_manifolds": 4}) == 10
        assert HiddenManifoldDiff.feature_dimension({"n_manifolds": 14}) == 10

    def test_feature_dimension_defaults(self):
        assert HiddenManifoldDiff.feature_dimension() == HiddenManifoldDiff.FEATURE_DIMENSION

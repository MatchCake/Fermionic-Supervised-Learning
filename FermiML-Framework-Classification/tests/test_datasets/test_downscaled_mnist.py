from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fermiml_framework_classification.datasets.downscaled_mnist import DownscaledMnist


def make_mock_ds(
    dimension: str,
    n_features: int = 20,
    n_train: int = DownscaledMnist.NUMBER_OF_TRAINING_SAMPLES,
    n_test: int = DownscaledMnist.NUMBER_OF_TEST_SAMPLES,
):
    rng = np.random.default_rng(0)
    ds = MagicMock()
    ds.train = {
        dimension: {
            "inputs": rng.standard_normal((n_train, n_features)),
            "labels": rng.choice([-1, 1], n_train),
        }
    }
    ds.test = {
        dimension: {
            "inputs": rng.standard_normal((n_test, n_features)),
            "labels": rng.choice([-1, 1], n_test),
        }
    }
    return ds


class TestDownscaledMnist:
    @pytest.fixture
    def mock_load(self):
        with patch("pennylane.data.load") as m:
            m.side_effect = lambda *a, **kw: [make_mock_ds("20")]
            yield m

    @pytest.fixture
    def dataset(self, mock_load):
        return DownscaledMnist(dimension=20, n_train_samples=250, n_test_samples=250, cv_splits=1)

    # --- init ---

    @pytest.mark.parametrize("d", [2, 5, 10, 20])
    def test_valid_dimension(self, d):
        ds = DownscaledMnist(dimension=d, cv_splits=1)
        assert ds._dimension == str(d)

    def test_dimension_as_string(self):
        ds = DownscaledMnist(dimension="10", cv_splits=1)
        assert ds._dimension == "10"

    def test_invalid_dimension_too_low(self):
        with pytest.raises(AssertionError):
            DownscaledMnist(dimension=1)

    def test_invalid_dimension_too_high(self):
        with pytest.raises(AssertionError):
            DownscaledMnist(dimension=21)

    def test_invalid_n_train_too_low(self):
        with pytest.raises(AssertionError):
            DownscaledMnist(n_train_samples=0)

    def test_invalid_n_train_too_high(self):
        with pytest.raises(AssertionError):
            DownscaledMnist(n_train_samples=DownscaledMnist.NUMBER_OF_TRAINING_SAMPLES + 1)

    def test_invalid_n_test_too_low(self):
        with pytest.raises(AssertionError):
            DownscaledMnist(n_test_samples=0)

    def test_invalid_n_test_too_high(self):
        with pytest.raises(AssertionError):
            DownscaledMnist(n_test_samples=DownscaledMnist.NUMBER_OF_TEST_SAMPLES + 1)

    def test_inherits_dataset(self):
        from fermiml_framework_classification.datasets.dataset import Dataset

        assert issubclass(DownscaledMnist, Dataset)

    # --- fetch methods ---

    def test_fetch_x_train_returns_array(self, dataset, mock_load):
        result = dataset.fetch_x_train()
        assert isinstance(result, np.ndarray)

    def test_fetch_y_train_returns_array(self, dataset, mock_load):
        result = dataset.fetch_y_train()
        assert isinstance(result, np.ndarray)

    def test_fetch_x_test_returns_array(self, dataset, mock_load):
        result = dataset.fetch_x_test()
        assert isinstance(result, np.ndarray)

    def test_fetch_y_test_returns_array(self, dataset, mock_load):
        result = dataset.fetch_y_test()
        assert isinstance(result, np.ndarray)

    def test_fetch_x_train_shape(self, dataset, mock_load):
        result = dataset.fetch_x_train()
        assert result.ndim == 2
        assert result.shape[0] == 250

    def test_fetch_x_test_shape(self, dataset, mock_load):
        result = dataset.fetch_x_test()
        assert result.ndim == 2
        assert result.shape[0] == 250

    def test_fetch_y_train_shape(self, dataset, mock_load):
        result = dataset.fetch_y_train()
        assert result.shape[0] == 250

    def test_fetch_y_test_shape(self, dataset, mock_load):
        result = dataset.fetch_y_test()
        assert result.shape[0] == 250

    def test_fetch_calls_load_with_correct_args(self, dataset, mock_load):
        dataset.fetch_x_train()
        args, kwargs = mock_load.call_args
        assert args == ("other",)
        assert kwargs["name"] == "downscaled-mnist"
        assert "folder_path" in kwargs

    def test_fetch_uses_dimension_key(self, mock_load):
        mock_load.side_effect = lambda *a, **kw: [make_mock_ds("5", n_features=5)]
        ds = DownscaledMnist(dimension=5, n_train_samples=10, n_test_samples=10, cv_splits=1)
        result = ds.fetch_x_train()
        assert result.shape[1] == 5

    def test_y_train_labels_are_binary(self, dataset, mock_load):
        result = dataset.fetch_y_train()
        assert set(np.unique(result)).issubset({0, 1})

    def test_y_test_labels_are_binary(self, dataset, mock_load):
        result = dataset.fetch_y_test()
        assert set(np.unique(result)).issubset({0, 1})

    def test_y_train_label_transform(self, mock_load):
        mock_load.side_effect = lambda *a, **kw: [make_mock_ds("20")]
        ds = DownscaledMnist(dimension=20, n_train_samples=250, n_test_samples=250, cv_splits=1)
        result = ds.fetch_y_train()
        assert np.all((result == 0) | (result == 1))

    def test_y_test_label_transform(self, mock_load):
        mock_load.side_effect = lambda *a, **kw: [make_mock_ds("20")]
        ds = DownscaledMnist(dimension=20, n_train_samples=250, n_test_samples=250, cv_splits=1)
        result = ds.fetch_y_test()
        assert np.all((result == 0) | (result == 1))

    def test_train_indexes_respect_n_train_samples(self, mock_load):
        n = 100
        ds = DownscaledMnist(dimension=20, n_train_samples=n, n_test_samples=50, cv_splits=1)
        assert len(ds._train_indexes) == n

    def test_test_indexes_respect_n_test_samples(self, mock_load):
        n = 50
        ds = DownscaledMnist(dimension=20, n_train_samples=100, n_test_samples=n, cv_splits=1)
        assert len(ds._test_indexes) == n

    def test_train_indexes_no_replacement(self, mock_load):
        ds = DownscaledMnist(dimension=20, n_train_samples=250, n_test_samples=250, cv_splits=1)
        assert len(ds._train_indexes) == len(set(ds._train_indexes))

    def test_test_indexes_no_replacement(self, mock_load):
        ds = DownscaledMnist(dimension=20, n_train_samples=250, n_test_samples=250, cv_splits=1)
        assert len(ds._test_indexes) == len(set(ds._test_indexes))

    def test_different_cv_split_id_gives_different_indexes(self, mock_load):
        ds0 = DownscaledMnist(dimension=20, n_train_samples=250, n_test_samples=250, cv_splits=2, cv_split_id=0)
        ds1 = DownscaledMnist(dimension=20, n_train_samples=250, n_test_samples=250, cv_splits=2, cv_split_id=1)
        assert not np.array_equal(ds0._train_indexes, ds1._train_indexes)

    # --- prepare_data / properties ---

    def test_x_train_property(self, dataset, mock_load):
        x = dataset.x_train
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2

    def test_y_train_length_matches_x_train(self, dataset, mock_load):
        assert len(dataset.y_train) == len(dataset.x_train)

    def test_x_val_property(self, dataset, mock_load):
        x = dataset.x_val
        assert isinstance(x, np.ndarray)
        assert x.ndim == 2

    def test_y_val_length_matches_x_val(self, dataset, mock_load):
        assert len(dataset.y_val) == len(dataset.x_val)

    def test_x_test_property(self, dataset, mock_load):
        x = dataset.x_test
        assert isinstance(x, np.ndarray)

    def test_train_val_sizes_sum_to_raw(self, dataset, mock_load):
        dataset.prepare_data()
        assert len(dataset._x_train) + len(dataset._x_val) == len(dataset._raw_x_train)

    def test_feature_dimension_follows_dimension(self):
        assert DownscaledMnist.feature_dimension({"dimension": 2}) == 2

    def test_feature_dimension_defaults(self):
        assert DownscaledMnist.feature_dimension() == DownscaledMnist.DEFAULT_DIMENSION

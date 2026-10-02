import numpy as np
import pytest

from fermiml_framework_regression.datasets.california_housing import CaliforniaHousingDataset

N_FEATURES = 8


def _fetchable() -> bool:
    """Whether scikit-learn can produce the dataset without reaching the network.

    The sweep populates a local cache on the login node precisely because compute nodes have no
    outbound internet. Mirroring that here keeps the suite runnable offline instead of failing
    on a download, and CI populates the cache on its first fetch.

    :return: True when the data is available.
    :rtype: bool
    """
    from sklearn.datasets import fetch_california_housing

    try:
        fetch_california_housing(download_if_missing=False)
        return True
    except Exception:  # noqa: BLE001 -- any failure means "not available here"
        try:
            fetch_california_housing()
            return True
        except Exception:  # noqa: BLE001
            return False


pytestmark = pytest.mark.skipif(not _fetchable(), reason="California Housing is neither cached nor downloadable")


class TestCaliforniaHousingDataset:
    """The one real-data set in the sweep, and the only one whose loader subsamples.

    It supplies 1240 of the sweep's jobs, so a mistake in the subsampling is not a crash but a
    whole dataset column of quietly wrong numbers.
    """

    @pytest.fixture(scope="class")
    def dataset(self):
        return CaliforniaHousingDataset(
            max_train_samples=120, max_test_samples=40, cv_splits=4, cv_split_id=0, val_size=0.25
        )

    def test_shapes(self, dataset):
        assert dataset.x_train.shape == (90, N_FEATURES)
        assert dataset.x_val.shape == (30, N_FEATURES)
        assert dataset.x_test.shape == (40, N_FEATURES)
        assert dataset.y_train.shape == (90,)
        assert dataset.y_test.shape == (40,)

    def test_dtypes_are_float32(self, dataset):
        for array in (dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test):
            assert array.dtype == np.float32

    def test_subsampling_caps_are_respected(self, dataset):
        assert len(dataset.x_train) + len(dataset.x_val) == 120
        assert len(dataset.x_test) == 40

    def test_train_and_test_pools_are_disjoint(self, dataset):
        train_rows = {row.tobytes() for row in dataset.x_train}
        assert not train_rows & {row.tobytes() for row in dataset.x_test}

    def test_train_and_val_partition_the_pool(self, dataset):
        train_rows = {row.tobytes() for row in dataset.x_train}
        val_rows = {row.tobytes() for row in dataset.x_val}
        assert not train_rows & val_rows
        assert len(train_rows | val_rows) == 120

    def test_targets_are_finite_and_positive(self, dataset):
        # Median house value in units of $100k; a non-positive or non-finite value would mean
        # the subsample indexed into the wrong array.
        assert np.all(np.isfinite(dataset.y_train))
        assert np.all(dataset.y_train > 0)

    def test_dataset_name_is_the_config_key(self):
        assert CaliforniaHousingDataset.DATASET_NAME == "california-housing"


class TestCaliforniaHousingReproducibility:
    def test_same_seed_gives_the_same_subsample(self):
        first = CaliforniaHousingDataset(max_train_samples=40, max_test_samples=10, random_state=0)
        second = CaliforniaHousingDataset(max_train_samples=40, max_test_samples=10, random_state=0)
        np.testing.assert_array_equal(first.x_train, second.x_train)
        np.testing.assert_array_equal(first.y_test, second.y_test)

    def test_different_seed_gives_a_different_subsample(self):
        first = CaliforniaHousingDataset(max_train_samples=40, max_test_samples=10, random_state=0)
        second = CaliforniaHousingDataset(max_train_samples=40, max_test_samples=10, random_state=5)
        assert not np.array_equal(first.x_train, second.x_train)

    def test_different_splits_of_one_pool_differ(self):
        first = CaliforniaHousingDataset(max_train_samples=40, max_test_samples=10, cv_splits=4, cv_split_id=0)
        second = CaliforniaHousingDataset(max_train_samples=40, max_test_samples=10, cv_splits=4, cv_split_id=1)
        # Same underlying pool and same test set, different train/validation partition.
        np.testing.assert_array_equal(first.x_test, second.x_test)
        assert not np.array_equal(first.x_val, second.x_val)

    def test_oversized_request_is_capped_at_eighty_percent(self):
        # max_train_samples above 80% of the data is clamped rather than silently overlapping
        # the test split.
        dataset = CaliforniaHousingDataset(max_train_samples=10**9, max_test_samples=10)
        assert len(dataset.x_train) + len(dataset.x_val) == int(0.8 * 20640)

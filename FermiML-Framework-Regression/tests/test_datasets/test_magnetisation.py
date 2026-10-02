import numpy as np
import pennylane as qml
import pytest
from matchcake.operations import MAngleEmbedding
from sklearn.metrics import r2_score

from fermiml_framework_regression.datasets.magnetisation import MagnetisationDataset
from fermiml_framework_regression.models.angle_nif_gp_regressor import AngleNIFGPRegressor
from tests.configs import ATOL_MATRIX_COMPARISON, MIN_RKHS_TEST_R2


def _statevector_magnetisation(x: np.ndarray, n_qubits: int, rotations: str = "Y,X") -> np.ndarray:
    """Total magnetisation from an exact state-vector simulation, as an oracle.

    :param x: Inputs of shape ``(n_samples, n_qubits)``.
    :param n_qubits: Number of qubits.
    :param rotations: Comma-separated rotation letters of the embedding.
    :return: Targets of shape ``(n_samples,)``.
    :rtype: np.ndarray
    """
    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device)
    def circuit(features):
        for rotation in rotations.split(","):
            MAngleEmbedding(features, wires=range(n_qubits), rotations=rotation)
        return qml.expval(qml.sum(*[qml.Z(k) for k in range(n_qubits)]))

    return np.array([float(circuit(row)) for row in x])


@pytest.mark.parametrize("n_qubits", [2, 4, 6])
class TestMagnetisationDataset:
    @pytest.fixture(scope="function")
    def dataset(self, n_qubits):
        return MagnetisationDataset(
            n_qubits=n_qubits,
            n_samples=40,
            n_test_samples=12,
            cv_splits=4,
            cv_split_id=0,
            val_size=0.25,
        )

    def test_shapes(self, dataset, n_qubits):
        assert dataset.x_train.shape == (30, n_qubits)
        assert dataset.y_train.shape == (30,)
        assert dataset.x_val.shape == (10, n_qubits)
        assert dataset.y_val.shape == (10,)
        assert dataset.x_test.shape == (12, n_qubits)
        assert dataset.y_test.shape == (12,)

    def test_dtypes_are_float32(self, dataset):
        for array in (dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test):
            assert array.dtype == np.float32

    def test_inputs_are_angles_in_range(self, dataset):
        assert dataset.x_train.min() >= 0.0
        assert dataset.x_train.max() <= np.pi

    def test_target_matches_statevector_oracle(self, dataset, n_qubits):
        expected = _statevector_magnetisation(dataset.x_test, n_qubits)
        np.testing.assert_allclose(dataset.y_test, expected, atol=ATOL_MATRIX_COMPARISON)

    def test_target_is_bounded_by_qubit_count(self, dataset, n_qubits):
        assert np.all(np.abs(dataset.y_test) <= n_qubits + ATOL_MATRIX_COMPARISON)

    def test_train_and_test_pools_are_disjoint(self, dataset):
        train_rows = {row.tobytes() for row in dataset.x_train}
        assert not train_rows & {row.tobytes() for row in dataset.x_test}

    def test_train_and_val_partition_the_pool(self, dataset):
        train_rows = {row.tobytes() for row in dataset.x_train}
        val_rows = {row.tobytes() for row in dataset.x_val}
        assert not train_rows & val_rows
        assert len(train_rows | val_rows) == 40


class TestMagnetisationDatasetConfiguration:
    def test_same_seed_is_reproducible(self):
        first = MagnetisationDataset(n_qubits=4, n_samples=16, n_test_samples=8)
        second = MagnetisationDataset(n_qubits=4, n_samples=16, n_test_samples=8)
        np.testing.assert_allclose(first.y_train, second.y_train)
        np.testing.assert_allclose(first.x_train, second.x_train)

    def test_different_seed_changes_the_data(self):
        first = MagnetisationDataset(n_qubits=4, n_samples=16, n_test_samples=8, random_state=0)
        second = MagnetisationDataset(n_qubits=4, n_samples=16, n_test_samples=8, random_state=7)
        assert not np.allclose(first.x_train, second.x_train)

    def test_noise_perturbs_train_but_not_test(self):
        clean = MagnetisationDataset(n_qubits=4, n_samples=32, n_test_samples=16, noise=0.0)
        noisy = MagnetisationDataset(n_qubits=4, n_samples=32, n_test_samples=16, noise=0.5)
        # Same seed, so the inputs are identical and only the targets may differ.
        np.testing.assert_allclose(clean.x_train, noisy.x_train)
        assert not np.allclose(clean.y_train, noisy.y_train)
        np.testing.assert_allclose(clean.y_test, noisy.y_test, atol=ATOL_MATRIX_COMPARISON)

    def test_odd_qubit_count_is_rejected(self):
        with pytest.raises(ValueError, match="even integer"):
            MagnetisationDataset(n_qubits=5)

    def test_too_few_qubits_is_rejected(self):
        with pytest.raises(ValueError, match="even integer"):
            MagnetisationDataset(n_qubits=0)

    def test_negative_noise_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            MagnetisationDataset(n_qubits=4, noise=-0.1)

    def test_dataset_name_is_the_config_key(self):
        assert MagnetisationDataset.DATASET_NAME == "magnetisation"


class TestMagnetisationIsInTheKernelRKHS:
    """The claim the dataset exists to test: a matched covariance GP fits it exactly."""

    def test_target_lies_in_the_span_of_the_feature_map(self):
        """The claim itself, tested directly rather than through a regularised GP.

        The target is a sum of two-point observables of the same matchgate embedding, so it is a
        *linear* functional of the order-2 Pfaffian features and an unregularised least-squares
        fit on them must reproduce it to numerical precision. Asserting it this way separates the
        RKHS property from everything a GP adds on top -- the jitter, the float32 kernel, the
        finite sample -- any of which can hide or fake it.

        This is the test that would have caught the kernel-cloning bug on its own: it is stated
        in the feature-map dimension, which is :math:`\\binom{2n}{2} = 28` at ``n_qubits=4`` and
        276 at the class default of 12, so a silently substituted kernel cannot satisfy it by
        overfitting instead.
        """
        dataset = MagnetisationDataset(n_qubits=4, n_samples=500, n_test_samples=200, cv_splits=4)
        model = AngleNIFGPRegressor(n_qubits=4, sector_m=2, rotations="Y,X")
        model.fit(dataset.x_train, dataset.y_train)
        assert model.nif_kernel_.n_terms == 28

        features_train = model.nif_kernel_.feature_map(dataset.x_train).detach().cpu().numpy().astype(np.float64)
        features_test = model.nif_kernel_.feature_map(dataset.x_test).detach().cpu().numpy().astype(np.float64)
        weights, *_ = np.linalg.lstsq(features_train, dataset.y_train.astype(np.float64), rcond=None)
        assert r2_score(dataset.y_test, features_test @ weights) > 0.999

    def test_matched_angle_gp_recovers_the_target(self):
        """The same claim through the actual model, which needs enough data to show it.

        Sized at 200 samples rather than 60: the matched kernel spans 28 dimensions, and a
        regularised GP over fewer than an order of magnitude more points than that does not
        resolve the target to :data:`MIN_RKHS_TEST_R2` (it reaches about 0.984 at 60). This used
        to pass at 60 only because the GP was silently fitting the 276-dimensional default-qubit
        kernel, which overfits that many points comfortably -- a flattering number for the wrong
        reason. See :class:`tests.test_sklearn_kernel_survives_cloning`.
        """
        dataset = MagnetisationDataset(n_qubits=4, n_samples=200, n_test_samples=100, cv_splits=4)
        model = AngleNIFGPRegressor(n_qubits=4, sector_m=2, rotations="Y,X")
        model.fit(dataset.x_train, dataset.y_train)
        assert r2_score(dataset.y_test, model.predict(dataset.x_test)) > MIN_RKHS_TEST_R2

    def test_model_with_fewer_qubits_than_features_raises(self):
        dataset = MagnetisationDataset(n_qubits=6, n_samples=20, n_test_samples=10)
        model = AngleNIFGPRegressor(n_qubits=4, sector_m=2)
        with pytest.raises(ValueError, match="Features must be of length"):
            model.fit(dataset.x_train, dataset.y_train)

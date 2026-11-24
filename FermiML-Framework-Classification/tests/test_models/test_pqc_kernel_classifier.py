import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from fermiml_framework_classification.experiments.single_experiment import (
    SingleExperiment,
)
from fermiml_framework_classification.models.pqc_encoding import PQCEncoding
from fermiml_framework_classification.models.pqc_kernel_classifier import (
    PQCKernelClassifier,
)
from tests.configs import (
    ATOL_APPROX_COMPARISON,
    ATOL_MATRIX_COMPARISON,
    ATOL_SINGLE_PRECISION_COMPARISON,
    TEST_SEED,
)


class TestPQCKernelClassifier:
    PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)

    @classmethod
    def _single_qubit_gate(cls, gate: np.ndarray, wire: int, n_qubits: int) -> np.ndarray:
        operator = np.eye(1, dtype=complex)
        for index in range(n_qubits):
            operator = np.kron(operator, gate if index == wire else np.eye(2, dtype=complex))
        return operator

    @classmethod
    def _cnot(cls, control: int, target: int, n_qubits: int) -> np.ndarray:
        projector_0 = np.diag([1.0, 0.0]).astype(complex)
        projector_1 = np.diag([0.0, 1.0]).astype(complex)
        return cls._single_qubit_gate(projector_0, control, n_qubits) + cls._single_qubit_gate(
            projector_1, control, n_qubits
        ) @ cls._single_qubit_gate(cls.PAULI_X, target, n_qubits)

    @classmethod
    def _reference_state(cls, scaled_x: np.ndarray, encoding: PQCEncoding) -> np.ndarray:
        """Simulate the circuit with plain matrices, independently of PennyLane and of the model's ansatz."""
        n_qubits = encoding.n_qubits
        angles = np.pi / 2 * encoding.reference_angles + np.pi / 2 * scaled_x
        state = np.zeros(2**n_qubits, dtype=complex)
        state[0] = 1.0
        for layer in range(encoding.depth):
            layer_angles = angles[layer * n_qubits : (layer + 1) * n_qubits]
            for wire, angle in enumerate(layer_angles):
                ry = np.array([[np.cos(angle / 2), -np.sin(angle / 2)], [np.sin(angle / 2), np.cos(angle / 2)]])
                state = cls._single_qubit_gate(ry.astype(complex), wire, n_qubits) @ state
            for wire, angle in enumerate(layer_angles):
                rz = np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
                state = cls._single_qubit_gate(rz, wire, n_qubits) @ state
            for first in range(layer % 2, n_qubits - 1, 2):
                state = cls._cnot(first, first + 1, n_qubits) @ state
        return state

    @classmethod
    def _reference_kernel(cls, model: PQCKernelClassifier, x0: np.ndarray, x_train: np.ndarray) -> np.ndarray:
        scaler = MinMaxScaler(feature_range=(0.0, 1.0)).fit(x_train)
        states0 = np.stack([cls._reference_state(row, model.encoding_) for row in scaler.transform(x0)])
        states1 = np.stack([cls._reference_state(row, model.encoding_) for row in scaler.transform(x_train)])
        return np.abs(states0.conj() @ states1.T) ** 2

    @pytest.fixture
    def x_train(self):
        return np.random.default_rng(TEST_SEED).uniform(-3.0, 5.0, (14, 8)).astype(np.float32)

    @pytest.fixture
    def x_test(self):
        return np.random.default_rng(TEST_SEED + 1).uniform(-3.0, 5.0, (6, 8)).astype(np.float32)

    @pytest.fixture
    def y_train(self):
        return np.array([0, 1] * 7).astype(int)

    @pytest.fixture
    def fitted(self, x_train, y_train):
        return PQCKernelClassifier(n_qubits=4, random_state=TEST_SEED).fit(x_train, y_train)

    def test_init(self):
        assert PQCKernelClassifier().MODEL_NAME == "PQC"

    def test_is_allowed_in_experiments(self):
        assert PQCKernelClassifier in SingleExperiment.ALLOWED_MODEL_CLS

    @pytest.mark.parametrize(
        "n_qubits, n_features, expected", [(-1, 6, 6), (-1, 5, 6), (4, 6, 4), (2, 6, 2), (8, 6, 6)]
    )
    def test_wires_follow_the_encoding(self, n_qubits, n_features, expected):
        rng = np.random.default_rng(TEST_SEED)
        model = PQCKernelClassifier(n_qubits=n_qubits).fit(rng.uniform(size=(8, n_features)), np.array([0, 1] * 4))
        assert len(model.wires) == expected

    @pytest.mark.parametrize("n_qubits", [4, 2, 8])
    def test_train_kernel_matches_reference(self, n_qubits, x_train, y_train):
        model = PQCKernelClassifier(n_qubits=n_qubits, random_state=TEST_SEED).fit(x_train, y_train)
        np.testing.assert_allclose(
            model.transform(x_train),
            self._reference_kernel(model, x_train, x_train),
            atol=ATOL_SINGLE_PRECISION_COMPARISON,
        )

    def test_rectangular_kernel_matches_reference_entry_by_entry(self, fitted, x_train, x_test):
        kernel = fitted.transform(x_test)
        assert kernel.shape == (6, 14)
        np.testing.assert_allclose(
            kernel, self._reference_kernel(fitted, x_test, x_train), atol=ATOL_SINGLE_PRECISION_COMPARISON
        )

    def test_rectangular_kernel_does_not_mark_test_points_as_training_points(self, fitted, x_test):
        leading_block = fitted.transform(x_test)[:6, :6]
        assert not np.any(np.isclose(np.diag(leading_block), 1.0, atol=ATOL_SINGLE_PRECISION_COMPARISON))
        assert not np.allclose(leading_block, leading_block.T, atol=ATOL_SINGLE_PRECISION_COMPARISON)

    def test_kernel_does_not_depend_on_gram_batch_size(self, x_train, y_train, x_test):
        one = PQCKernelClassifier(n_qubits=4, gram_batch_size=1, random_state=TEST_SEED).fit(x_train, y_train)
        many = PQCKernelClassifier(n_qubits=4, gram_batch_size=5, random_state=TEST_SEED).fit(x_train, y_train)
        np.testing.assert_allclose(one.transform(x_test), many.transform(x_test), atol=ATOL_SINGLE_PRECISION_COMPARISON)

    def test_train_kernel_is_a_valid_kernel(self, fitted, x_train):
        kernel = np.asarray(fitted.transform(x_train), dtype=np.float64)
        np.testing.assert_allclose(kernel, kernel.T, atol=ATOL_SINGLE_PRECISION_COMPARISON)
        np.testing.assert_allclose(np.diag(kernel), 1.0, atol=ATOL_SINGLE_PRECISION_COMPARISON)
        assert np.linalg.eigvalsh((kernel + kernel.T) / 2).min() > -ATOL_SINGLE_PRECISION_COMPARISON

    def test_predict_matches_an_svm_on_the_reference_kernel(self, fitted, x_train, x_test, y_train):
        reference = SVC(kernel="precomputed", C=fitted.svm_c).fit(
            self._reference_kernel(fitted, x_train, x_train), y_train
        )
        np.testing.assert_array_equal(
            fitted.predict(x_test), reference.predict(self._reference_kernel(fitted, x_test, x_train))
        )

    def test_predict_proba(self, fitted, x_test):
        probabilities = fitted.predict_proba(x_test)
        assert probabilities.shape == (6, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=ATOL_MATRIX_COMPARISON)
        np.testing.assert_allclose(
            probabilities,
            fitted.svm.predict_proba(self._reference_kernel(fitted, x_test, fitted.x_train_)),
            atol=ATOL_APPROX_COMPARISON,
        )

    @pytest.mark.parametrize("n_features, n_qubits", [(11, 4), (13, 4), (7, 2), (5, 4)])
    def test_partial_last_layer_matches_reference(self, n_features, n_qubits, y_train):
        x_train = np.random.default_rng(TEST_SEED).uniform(-1.0, 1.0, (14, n_features)).astype(np.float32)
        model = PQCKernelClassifier(n_qubits=n_qubits, random_state=TEST_SEED).fit(x_train, y_train)
        np.testing.assert_allclose(
            model.transform(x_train),
            self._reference_kernel(model, x_train, x_train),
            atol=ATOL_SINGLE_PRECISION_COMPARISON,
        )

    def test_refit_on_another_dimension_rebuilds_the_circuit(self, fitted):
        rng = np.random.default_rng(TEST_SEED)
        fitted.fit(rng.uniform(size=(8, 6)), np.array([0, 1] * 4))
        assert len(fitted.wires) == 4
        assert fitted.encoding_.depth == 2
        assert fitted.predict(rng.uniform(size=(3, 6))).shape == (3,)

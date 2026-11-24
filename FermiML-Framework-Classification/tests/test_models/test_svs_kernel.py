import numpy as np
import pennylane as qml
import pytest
from matchcake.utils.torch_utils import to_numpy

from fermiml_framework_classification.models.iqp_kernel_classifier import (
    IQPKernelClassifier,
)
from fermiml_framework_classification.models.svs_kernel import SVSKernel
from tests.configs import ATOL_SINGLE_PRECISION_COMPARISON, TEST_SEED


class TestSVSKernel:
    @classmethod
    def _exact_kernel(cls, model: IQPKernelClassifier, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """Evaluate the fidelity with a separate adjoint circuit for each pair, sharing no code with the kernel."""
        n_qubits = x0.shape[-1]

        @qml.qnode(qml.device("default.qubit", wires=n_qubits))
        def zero_probability(first, second):
            qml.IQPEmbedding(first, wires=range(n_qubits), n_repeats=model.repeats)
            qml.adjoint(qml.IQPEmbedding)(second, wires=range(n_qubits), n_repeats=model.repeats)
            return qml.probs(wires=range(n_qubits))

        scaled0, scaled1 = model.scaler.transform(x0), model.scaler.transform(x1)
        return np.array([[float(zero_probability(row, col)[0]) for col in scaled1] for row in scaled0])

    @pytest.fixture
    def model(self):
        return SVSKernel()

    @pytest.fixture
    def x_train(self):
        return np.random.default_rng(TEST_SEED).uniform(-3.0, 5.0, (8, 4)).astype(np.float32)

    @pytest.fixture
    def fitted(self, x_train):
        return IQPKernelClassifier(random_state=TEST_SEED).fit(x_train, np.array([0, 1] * 4))

    def test_init(self, model):
        assert model.MODEL_NAME == "SVSKernel"

    @pytest.mark.parametrize("n_points", [3, 8, 11])
    def test_kernel_between_different_points_is_exact(self, fitted, x_train, n_points):
        others = np.random.default_rng(TEST_SEED + 1).uniform(-3.0, 5.0, (n_points, 4)).astype(np.float32)
        kernel = to_numpy(fitted.transform(others))
        assert kernel.shape == (n_points, 8)
        np.testing.assert_allclose(
            kernel, self._exact_kernel(fitted, others, x_train), atol=ATOL_SINGLE_PRECISION_COMPARISON
        )

    def test_kernel_between_different_points_has_no_spurious_identity(self, fitted):
        others = np.random.default_rng(TEST_SEED + 1).uniform(-3.0, 5.0, (3, 4)).astype(np.float32)
        assert not np.any(
            np.isclose(np.diag(to_numpy(fitted.transform(others))), 1.0, atol=ATOL_SINGLE_PRECISION_COMPARISON)
        )

    def test_training_kernel_is_exact_symmetric_and_has_a_unit_diagonal(self, fitted, x_train):
        kernel = to_numpy(fitted.transform(x_train))
        np.testing.assert_array_equal(kernel, kernel.T)
        np.testing.assert_array_equal(np.diag(kernel), 1.0)
        np.testing.assert_allclose(
            kernel, self._exact_kernel(fitted, x_train, x_train), atol=ATOL_SINGLE_PRECISION_COMPARISON
        )

    @pytest.mark.parametrize("n_points", [3, 8])
    def test_simulates_the_states_in_the_original_order(self, fitted, x_train, n_points, monkeypatch):
        # QKS samples from one seeded stream, so the number and order of simulations must not change.
        calls = []
        original = fitted._x_to_sv
        monkeypatch.setattr(fitted, "_x_to_sv", lambda x: (calls.append(int(x.shape[0])), original(x))[1])
        fitted.__dict__.pop("sv_train", None)
        others = x_train[:n_points]
        fitted.transform(others)
        assert calls == [n_points, 8]
        calls.clear()
        fitted.transform(x_train)
        assert calls == [8]

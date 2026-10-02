import numpy as np
import pytest
from matchcake.operations.state_preparation.product_state import ProductState
from matchcake.utils.torch_utils import to_tensor

from fermiml_framework_classification.models.fermi_vanilla_kernel_classifier import (
    FermiVanillaKernelClassifier,
)
from fermiml_framework_classification.models.fermi_vanilla_plus_kernel_classifier import (
    FermiVanillaPlusKernelClassifier,
)
from tests.configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON


class TestFermiVanillaPlusKernelClassifier:
    @pytest.fixture
    def model(self):
        return FermiVanillaPlusKernelClassifier()

    @pytest.fixture
    def x_train(self):
        return np.arange(30).reshape(3, 10).astype(np.float32)

    @pytest.fixture
    def y_train(self):
        return np.array([0, 1, 0]).astype(int)

    def test_init(self, model):
        assert model.MODEL_NAME == "Vanilla+"

    def test_is_subclass_of_vanilla(self, model):
        assert isinstance(model, FermiVanillaKernelClassifier)

    def test_kernel_is_asymmetric(self, x_train, y_train):
        model = FermiVanillaPlusKernelClassifier(n_qubits=4, random_state=0)
        model.fit(x_train, y_train)
        kernel_matrix = model.transform(x_train)
        assert not np.allclose(kernel_matrix, kernel_matrix.T, atol=ATOL_MATRIX_COMPARISON)

    def test_diagonal_is_uniform_transition_probability(self, x_train, y_train):
        model = FermiVanillaPlusKernelClassifier(n_qubits=4, random_state=0)
        model.fit(x_train, y_train)
        kernel_matrix = model.transform(x_train)
        expected = 1.0 / (2**model.n_qubits)
        np.testing.assert_allclose(
            np.diag(kernel_matrix),
            np.full(len(x_train), expected),
            atol=ATOL_MATRIX_COMPARISON,
            rtol=RTOL_MATRIX_COMPARISON,
        )

    def test_predict(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        predictions = model.predict(x_train)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (3,)

    def test_predict_proba(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        predictions = model.predict_proba(x_train)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (3, 2)

    def test_fit_sets_n_qubits(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        assert model.n_qubits == x_train.shape[-1]

    def test_circuit_prepends_plus_product_state(self, x_train, y_train):
        model = FermiVanillaPlusKernelClassifier(n_qubits=4)
        model.fit(x_train, y_train)
        x = to_tensor(model.scaler.transform(x_train), dtype=model.R_DTYPE)
        sptm = model._x_to_sptm(x)
        ops = list(model.circuit(sptm[0], sptm[0]))
        assert isinstance(ops[0], ProductState)
        plus = np.array([1.0, 1.0]) / np.sqrt(2)
        np.testing.assert_allclose(np.real(np.asarray(ops[0].data[0])), np.tile(plus, (model.n_qubits, 1)))

    def test_plus_kernel_differs_from_vacuum(self, x_train, y_train):
        plus = FermiVanillaPlusKernelClassifier(n_qubits=4, random_state=0).fit(x_train, y_train)
        vacuum = FermiVanillaKernelClassifier(n_qubits=4, random_state=0).fit(x_train, y_train)
        plus_kernel = plus.transform(x_train)
        vacuum_kernel = vacuum.transform(x_train)
        assert not np.allclose(plus_kernel, vacuum_kernel)

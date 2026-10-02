import numpy as np
import pytest

from fermiml_framework_classification.models.fermi_vanilla_kernel_classifier import (
    FermiVanillaKernelClassifier,
)


class TestFermiVanillaKernelClassifier:
    @pytest.fixture
    def model(self):
        return FermiVanillaKernelClassifier()

    @pytest.fixture
    def x_train(self):
        return np.arange(30).reshape(3, 10).astype(np.float32)

    @pytest.fixture
    def y_train(self):
        return np.array([0, 1, 0]).astype(int)

    def test_init(self, model):
        assert model.MODEL_NAME == "Vanilla"

    def test_kernel_symmetry(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        kernel_matrix = model.transform(x_train)
        np.testing.assert_allclose(kernel_matrix, kernel_matrix.T)

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

    def test_explicit_n_qubits_skips_default_flag(self):
        model = FermiVanillaKernelClassifier(n_qubits=4)
        assert model._n_qubits_is_default_flag is False
        assert model.n_qubits == 4

    def test_fit_preserves_explicit_n_qubits(self, x_train, y_train):
        model = FermiVanillaKernelClassifier(n_qubits=4)
        model.fit(x_train, y_train)
        assert model.n_qubits == 4

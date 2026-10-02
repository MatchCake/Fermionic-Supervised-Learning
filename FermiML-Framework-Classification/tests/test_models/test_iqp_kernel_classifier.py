import numpy as np
import pennylane as qml
import pytest

from fermiml_framework_classification.models.iqp_kernel_classifier import (
    IQPKernelClassifier,
)


class TestIQPKernelClassifier:
    @pytest.fixture
    def model(self):
        return IQPKernelClassifier()

    @pytest.fixture
    def x_train(self):
        return np.arange(30).reshape(3, 10).astype(np.float32)

    @pytest.fixture
    def y_train(self):
        return np.array([0, 1, 0]).astype(int)

    def test_init(self, model):
        assert model.MODEL_NAME == "IQP"

    def test_wires_property(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        assert len(model.wires) == x_train.shape[-1]

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

    def test_circuit_from_probs(self, model, x_train, y_train):
        model.repeats = 2
        model.fit(x_train, y_train)

        @qml.qnode(qml.device("default.qubit", wires=model.n_qubits))
        def xanadu_circuit(x0, x1):
            """
            Implementation taken from: https://github.com/XanaduAI/qml-benchmarks/blob/95e5a07e8e9e75ba7e24e67fb32b030112a1309a/src/qml_benchmarks/models/iqp_kernel.py#L95
            """
            qml.IQPEmbedding(x0, wires=range(model.n_qubits), n_repeats=model.repeats)
            qml.adjoint(
                qml.IQPEmbedding(
                    x1,
                    wires=range(model.n_qubits),
                    n_repeats=model.repeats,
                )
            )
            return qml.probs()

        scaled_x_train = model.scaler.transform(x_train)
        xanadu_kernel_matrix = np.asarray(
            [
                xanadu_circuit(scaled_x_train[i], scaled_x_train[j])[0]
                for i in range(len(x_train))
                for j in range(len(x_train))
            ]
        ).reshape(len(x_train), len(x_train))
        np.testing.assert_allclose(model.transform(x_train), xanadu_kernel_matrix, atol=1e-5)

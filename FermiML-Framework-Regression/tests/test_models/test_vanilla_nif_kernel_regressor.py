import numpy as np
import pytest
from matchcake.utils.torch_utils import to_numpy

from fermiml_framework_regression.models.vanilla_nif_kernel_regressor import VanillaNIFKernelRegressor
from tests.configs import ATOL_MATRIX_COMPARISON, DEFAULT_SEED


@pytest.fixture
def training_data():
    rng = np.random.default_rng(DEFAULT_SEED)
    x = rng.uniform(-1.0, 1.0, size=(24, 6)).astype(np.float32)
    return x, x.sum(axis=-1).astype(np.float32)


class TestVanillaNIFKernelRegressorAutoQubits:
    """``n_qubits=-1`` is the setting the whole config grid uses, so it has to construct."""

    def test_auto_qubits_constructs(self):
        # This class *is* the kernel, so the device is built in __init__ rather than at fit
        # time; passing -1 straight through used to assert before fit could resolve it.
        model = VanillaNIFKernelRegressor(n_qubits=-1)
        assert model.n_qubits == VanillaNIFKernelRegressor.PLACEHOLDER_N_QUBITS

    def test_auto_qubits_resolves_to_the_feature_count_on_fit(self, training_data):
        x, y = training_data
        model = VanillaNIFKernelRegressor(n_qubits=-1)
        model.fit(x, y)
        assert model.n_qubits == x.shape[-1]

    def test_auto_qubits_predicts_the_right_shape(self, training_data):
        x, y = training_data
        model = VanillaNIFKernelRegressor(n_qubits=-1)
        model.fit(x, y)
        assert model.predict(x).shape == (len(x),)

    def test_explicit_qubits_are_untouched(self, training_data):
        x, y = training_data
        model = VanillaNIFKernelRegressor(n_qubits=8)
        assert model.n_qubits == 8
        model.fit(x, y)
        # Still 8, not re-derived from the 6 input features.
        assert model.n_qubits == 8

    def test_fidelity_gram_has_unit_diagonal(self, training_data):
        x, y = training_data
        model = VanillaNIFKernelRegressor(n_qubits=-1)
        model.fit(x, y)
        gram = to_numpy(model.transform(model._scale_x(x)))
        np.testing.assert_allclose(np.diag(gram), 1.0, atol=ATOL_MATRIX_COMPARISON)

import numpy as np
import pytest
import torch

from fermiml_framework_classification.models.fermi_angle_mpf_kernel_classifier import (
    FermiAngleMpfKernelClassifier,
)
from fermiml_framework_classification.models.fermi_vanilla_mpf_kernel_classifier import (
    FermiVanillaMpfKernelClassifier,
)
from fermiml_framework_classification.models.fermi_vanilla_mpf_plus_kernel_classifier import (
    FermiVanillaMpfPlusKernelClassifier,
)
from fermiml_framework_classification.models.mpf_normalization import (
    HILBERT_DIMENSION_NORM,
    kernel_prefactor,
)
from tests.configs import ATOL_MATRIX_COMPARISON, ATOL_SCALAR_COMPARISON

MPF_CLASSES = [FermiVanillaMpfKernelClassifier, FermiVanillaMpfPlusKernelClassifier, FermiAngleMpfKernelClassifier]


class TestMpfNormalization:
    @pytest.mark.parametrize("n_qubits", [2, 10, 64])
    def test_numeric_norm_keeps_the_exponential_prefactor(self, n_qubits):
        expected = 3.0 / 2.0**n_qubits
        np.testing.assert_allclose(kernel_prefactor(3.0, n_qubits), expected, atol=ATOL_SCALAR_COMPARISON)

    @pytest.mark.parametrize("n_qubits", [2, 10, 64, 2000])
    def test_hilbert_dimension_norm_cancels_the_prefactor_at_any_size(self, n_qubits):
        assert kernel_prefactor(HILBERT_DIMENSION_NORM, n_qubits) == 1.0

    @pytest.mark.parametrize("norm", ["2**n", "", "full", None])
    def test_unknown_norm_raises(self, norm):
        with pytest.raises(ValueError):
            kernel_prefactor(norm, 4)

    @pytest.mark.parametrize("model_cls", MPF_CLASSES)
    def test_unknown_norm_raises_at_construction(self, model_cls):
        with pytest.raises(ValueError):
            model_cls(observable_norm="2**n")

    @pytest.mark.parametrize("model_cls", MPF_CLASSES)
    @pytest.mark.parametrize("n_qubits", [-1, 8])
    def test_string_norm_rescales_every_kernel_block_by_two_to_the_n(self, model_cls, n_qubits):
        rng = np.random.default_rng(0)
        x_train, x_test = rng.uniform(size=(12, 4)), rng.uniform(size=(5, 4))
        y_train = np.array([0, 1] * 6)
        models = []
        for norm in (1.0, HILBERT_DIMENSION_NORM):
            torch.manual_seed(0)  # the encoder weights come from the global generator
            models.append(model_cls(random_state=0, n_qubits=n_qubits, observable_norm=norm).fit(x_train, y_train))
        unit, rescaled = models
        assert unit.n_qubits == rescaled.n_qubits
        factor = 2.0**unit.n_qubits
        for x in (x_train, x_test):
            np.testing.assert_allclose(
                np.asarray(rescaled.transform(x)),
                np.asarray(unit.transform(x)) * factor,
                atol=ATOL_MATRIX_COMPARISON,
                rtol=1e-5,
            )

    @pytest.mark.parametrize("model_cls", MPF_CLASSES)
    def test_rescaled_kernel_fits_where_the_unit_norm_collapses(self, model_cls):
        rng = np.random.default_rng(0)
        x_train = rng.uniform(-1.0, 1.0, size=(60, 16))
        y_train = (x_train[:, 0] > 0).astype(int)
        unit = model_cls(random_state=0, svm_c=1.0, n_qubits=16, observable_norm=1.0).fit(x_train, y_train)
        rescaled = model_cls(random_state=0, svm_c=1.0, n_qubits=16, observable_norm=HILBERT_DIMENSION_NORM).fit(
            x_train, y_train
        )
        assert np.unique(unit.predict(x_train)).size == 1
        assert np.unique(rescaled.predict(x_train)).size == 2

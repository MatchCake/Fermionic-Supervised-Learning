from itertools import combinations
from math import comb

import numpy as np
import pytest
import torch

from fermiml_framework_regression.angle_mpf_plus_covariance_kernel import AngleMpfPlusCovarianceKernel
from fermiml_framework_regression.angle_nif_covariance_kernel import AngleNIFCovarianceKernel
from fermiml_framework_regression.experiments.single_experiment import SingleExperiment
from fermiml_framework_regression.models.angle_mpf_plus_gp_regressor import AngleMpfPlusGPRegressor
from fermiml_framework_regression.models.vanilla_mpf_plus_gp_regressor import VanillaMpfPlusGPRegressor
from fermiml_framework_regression.vanilla_mpf_plus_covariance_kernel import VanillaMpfPlusCovarianceKernel
from fermiml_framework_regression.vanilla_nif_covariance_kernel import VanillaNIFCovarianceKernel

KERNELS = {
    "AngleMpfPlus": (AngleMpfPlusCovarianceKernel, AngleNIFCovarianceKernel),
    "VanillaMpfPlus": (VanillaMpfPlusCovarianceKernel, VanillaNIFCovarianceKernel),
}
REGRESSORS = {
    "AngleMpfPlus": AngleMpfPlusGPRegressor,
    "VanillaMpfPlus": VanillaMpfPlusGPRegressor,
}
N_QUBITS = 4


@pytest.fixture
def training_data():
    rng = np.random.RandomState(0)
    x = rng.uniform(0.0, 1.0, size=(24, N_QUBITS)).astype(np.float32)
    y = np.sin(3.0 * x[:, 0]).astype(np.float32)
    return x, y


@pytest.mark.parametrize("kernel_name", sorted(KERNELS))
class TestMultiSectorFeatureMap:
    """The feature map spans several Majorana orders, read off the parity-extended algebra."""

    def test_dimension_is_the_sum_over_orders(self, kernel_name, training_data):
        x, y = training_data
        kernel_cls, _ = KERNELS[kernel_name]
        for sectors in ((1, 2, 3), (1, 2, 3, 4), (2,)):
            torch.manual_seed(0)
            kernel = kernel_cls(sectors=sectors, n_qubits=N_QUBITS)
            kernel.fit(x, y)
            expected = sum(comb(2 * N_QUBITS, order) for order in sectors)
            assert kernel.n_terms == expected
            assert kernel.feature_map(x).shape == (len(x), expected)

    def test_odd_orders_are_closed_with_the_parity_index(self, kernel_name, training_data):
        """An odd-order subset spans an odd-dimensional skew matrix, whose Pfaffian is undefined."""
        x, y = training_data
        kernel_cls, _ = KERNELS[kernel_name]
        torch.manual_seed(0)
        kernel = kernel_cls(sectors=(1, 2, 3), n_qubits=N_QUBITS)
        kernel.fit(x, y)
        for order, indices in zip(kernel.orders, kernel.index_sets_per_order):
            assert indices.shape == (comb(2 * N_QUBITS, order), order + (order % 2))
            if order % 2:
                assert np.all(indices[:, -1] == kernel.parity_index)
                # The parity index closes the subset; it must not also appear inside it.
                assert np.all(indices[:, :-1] < kernel.parity_index)
            else:
                assert np.all(indices < kernel.parity_index)

    def test_orders_are_sorted_and_deduplicated(self, kernel_name):
        kernel_cls, _ = KERNELS[kernel_name]
        torch.manual_seed(0)
        assert kernel_cls(sectors=[3, 1, 2, 1], n_qubits=N_QUBITS).orders == (1, 2, 3)

    def test_sectors_are_stored_verbatim(self, kernel_name):
        """sklearn's clone() compares constructor arguments by identity, and the GP clones its kernel."""
        kernel_cls, _ = KERNELS[kernel_name]
        torch.manual_seed(0)
        sectors = [1, 2, 3]
        assert kernel_cls(sectors=sectors, n_qubits=N_QUBITS).sectors is sectors

    def test_rejects_an_order_beyond_its_bound(self, kernel_name, training_data):
        """Even orders are bounded by 2n-2; an odd order closed with parity reaches 2n."""
        x, y = training_data
        kernel_cls, _ = KERNELS[kernel_name]
        torch.manual_seed(0)
        kernel = kernel_cls(sectors=(2 * N_QUBITS,), n_qubits=N_QUBITS)
        with pytest.raises(ValueError, match="exceeds"):
            _ = kernel.index_sets_per_order
        torch.manual_seed(0)
        # 2n is admissible for an odd order only, so 2n-1 must be accepted.
        assert kernel_cls(sectors=(2 * N_QUBITS - 1,), n_qubits=N_QUBITS).n_terms == comb(
            2 * N_QUBITS, 2 * N_QUBITS - 1
        )

    def test_rejects_an_empty_or_nonpositive_order_set(self, kernel_name):
        kernel_cls, _ = KERNELS[kernel_name]
        for sectors in ((), (0, 1), (-2,)):
            torch.manual_seed(0)
            with pytest.raises(ValueError, match="Majorana orders"):
                kernel_cls(sectors=sectors, n_qubits=N_QUBITS)

    def test_odd_orders_need_the_plus_state(self, kernel_name, training_data):
        """The premise the whole construction rests on, measured rather than assumed.

        The displacement vector :math:`d_\\mu = \\langle c_\\mu \\rangle` is what makes an odd-order
        Pfaffian nonzero, and it vanishes on a computational basis state. Starting from the
        fermionic vacuum instead of :math:`\\ket{+}^{\\otimes n}` therefore zeroes every odd block,
        which would make the extra orders free parameters carrying nothing.
        """
        x, y = training_data
        kernel_cls, base_cls = KERNELS[kernel_name]

        class VacuumVariant(kernel_cls):  # type: ignore[valid-type,misc]
            def ansatz(self, batch):
                # Skip the |+> preparation, keep the feature map and everything else.
                yield from base_cls.ansatz(self, batch)

        magnitudes = []
        for cls in (VacuumVariant, kernel_cls):
            torch.manual_seed(0)
            kernel = cls(sectors=(1, 3), n_qubits=N_QUBITS)
            kernel.fit(x, y)
            magnitudes.append(float(torch.max(torch.abs(kernel.feature_map(x)))))
        vacuum, plus = magnitudes
        assert vacuum == pytest.approx(0.0, abs=1e-6)
        assert plus > 1e-3


@pytest.mark.parametrize("model_name", sorted(REGRESSORS))
class TestMpfPlusRegressors:
    """The regressors that wrap those kernels."""

    def test_fit_predict_runs_and_matches_the_kernel_dimension(self, model_name, training_data):
        x, y = training_data
        torch.manual_seed(0)
        model = REGRESSORS[model_name](n_qubits=-1, sectors=(1, 2, 3), alpha=1e-4)
        model.fit(x, y)
        predictions = model.predict(x)
        assert predictions.shape == y.shape
        assert np.all(np.isfinite(predictions))
        assert model.nif_kernel_.n_terms == sum(comb(2 * N_QUBITS, order) for order in (1, 2, 3))

    def test_accepts_a_yaml_list_of_sectors(self, model_name, training_data):
        """A sector set read from the experiment config arrives as a list, not a tuple."""
        x, y = training_data
        torch.manual_seed(0)
        model = REGRESSORS[model_name](n_qubits=-1, sectors=[1, 2, 3, 4], alpha=1e-4)
        model.fit(x, y)
        assert model.nif_kernel_.orders == (1, 2, 3, 4)

    def test_repeated_fits_agree(self, model_name, training_data):
        """Results are addressed by a config hash, so a requeued job must not change its answer."""
        x, y = training_data
        predictions = []
        for _ in range(2):
            model = REGRESSORS[model_name](n_qubits=-1, sectors=(1, 2, 3), alpha=1e-4)
            model.fit(x, y)
            predictions.append(model.predict(x))
        np.testing.assert_allclose(predictions[0], predictions[1], rtol=0, atol=0)

    def test_is_registered_and_resolvable_by_name(self, model_name):
        """SingleExperiment resolves a config's model by MODEL_NAME, case-insensitively."""
        model_cls = REGRESSORS[model_name]
        assert model_cls in SingleExperiment.ALLOWED_MODEL_CLS
        assert model_cls.MODEL_NAME == model_name
        matches = [cls for cls in SingleExperiment.ALLOWED_MODEL_CLS if cls.MODEL_NAME.lower() == model_name.lower()]
        assert matches == [model_cls]


def test_the_two_kernels_differ_only_in_their_ansatz(training_data):
    """Same orders, same qubit count, same feature dimension: only the embedding differs."""
    x, y = training_data
    features = {}
    for name, (kernel_cls, _) in KERNELS.items():
        torch.manual_seed(0)
        kernel = kernel_cls(sectors=(1, 2, 3), n_qubits=N_QUBITS)
        kernel.fit(x, y)
        features[name] = kernel.feature_map(x)
    assert features["AngleMpfPlus"].shape == features["VanillaMpfPlus"].shape
    assert not torch.allclose(features["AngleMpfPlus"], features["VanillaMpfPlus"])


def test_index_sets_are_rebuilt_when_the_qubit_count_changes(training_data):
    """The per-order arrays are keyed on n_qubits and must not survive a change to it."""
    torch.manual_seed(0)
    kernel = AngleMpfPlusCovarianceKernel(sectors=(1, 2), n_qubits=4)
    assert kernel.n_terms == comb(8, 1) + comb(8, 2)
    kernel.n_qubits = 6
    assert kernel.n_terms == comb(12, 1) + comb(12, 2)
    assert kernel.parity_index == 12
    expected_first = np.array([subset + (12,) for subset in combinations(range(12), 1)], dtype=np.int64)
    np.testing.assert_array_equal(kernel.index_sets_per_order[0], expected_first)

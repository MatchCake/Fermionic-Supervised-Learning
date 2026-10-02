import numpy as np
import pytest
from matchcake.operations.state_preparation.product_state import ProductState
from matchcake.utils.torch_utils import to_tensor

from fermiml_framework_classification.models.fermi_vanilla_mpf_kernel_classifier import (
    FermiVanillaMpfKernelClassifier,
)
from fermiml_framework_classification.models.fermi_vanilla_mpf_plus_kernel_classifier import (
    FermiVanillaMpfPlusKernelClassifier,
)
from tests.configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON


class TestFermiVanillaMpfPlusKernelClassifier:
    @pytest.fixture
    def model(self):
        return FermiVanillaMpfPlusKernelClassifier()

    @pytest.fixture
    def x_train(self):
        return np.arange(30).reshape(3, 10).astype(np.float32)

    @pytest.fixture
    def y_train(self):
        return np.array([0, 1, 0]).astype(int)

    def test_init(self, model):
        assert model.MODEL_NAME == "VanillaMpf+"

    def test_is_subclass_of_vanilla_mpf(self, model):
        assert isinstance(model, FermiVanillaMpfKernelClassifier)

    def test_kernel_symmetry(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        kernel_matrix = model.transform(x_train)
        np.testing.assert_allclose(
            kernel_matrix, kernel_matrix.T, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
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

    def test_sector_m_default(self, model):
        assert model.sector_m == FermiVanillaMpfPlusKernelClassifier.DEFAULT_SECTOR_M

    def test_ansatz_prepends_plus_product_state(self, x_train, y_train):
        model = FermiVanillaMpfPlusKernelClassifier(n_qubits=4)
        model.fit(x_train, y_train)
        x = to_tensor(model.scaler.transform(x_train), dtype=model.R_DTYPE)
        ops = list(model.ansatz(x))
        assert isinstance(ops[0], ProductState)
        plus = np.array([1.0, 1.0]) / np.sqrt(2)
        np.testing.assert_allclose(np.real(np.asarray(ops[0].data[0])), np.tile(plus, (model.n_qubits, 1)))

    def test_plus_feature_map_differs_from_vacuum(self, x_train, y_train):
        plus = FermiVanillaMpfPlusKernelClassifier(n_qubits=4, random_state=0).fit(x_train, y_train)
        vacuum = FermiVanillaMpfKernelClassifier(n_qubits=4, random_state=0).fit(x_train, y_train)
        plus_kernel = plus.transform(x_train)
        vacuum_kernel = vacuum.transform(x_train)
        assert not np.allclose(plus_kernel, vacuum_kernel)

    def test_sectors_defaults_to_none_and_reports_the_single_order(self, model):
        assert model.uses_extended_algebra is False
        assert model.sectors == (model.sector_m,)

    def test_sectors_spans_the_requested_orders(self):
        model = FermiVanillaMpfPlusKernelClassifier(n_qubits=4, sectors=[2, 3])
        assert model.uses_extended_algebra is True
        assert model.sectors == (2, 3)

    def test_sectors_is_sorted_and_deduplicated_on_input(self):
        assert FermiVanillaMpfPlusKernelClassifier(n_qubits=4, sectors=[3, 2]).sectors == (2, 3)

    def test_sectors_rejects_a_repeated_order(self):
        with pytest.raises(ValueError, match="distinct"):
            FermiVanillaMpfPlusKernelClassifier(n_qubits=4, sectors=[2, 2])

    def test_sectors_rejects_an_empty_specification(self):
        with pytest.raises(ValueError, match="at least one"):
            FermiVanillaMpfPlusKernelClassifier(n_qubits=4, sectors=[])

    def test_sectors_rejects_a_non_positive_order(self):
        with pytest.raises(ValueError, match=">= 1"):
            FermiVanillaMpfPlusKernelClassifier(n_qubits=4, sectors=[0, 2])

    def test_odd_order_index_sets_are_closed_with_the_parity_index(self):
        model = FermiVanillaMpfPlusKernelClassifier(n_qubits=4, sectors=[3])
        (indices,) = model.index_sets_per_order
        # An odd subset gains the parity index so its submatrix is even-dimensional and its Pfaffian defined.
        assert indices.shape[1] == 4
        assert (indices[:, -1] == model.parity_index).all()

    def test_even_order_index_sets_are_untouched(self):
        model = FermiVanillaMpfPlusKernelClassifier(n_qubits=4, sectors=[2])
        (indices,) = model.index_sets_per_order
        assert indices.shape[1] == 2
        assert (indices < model.parity_index).all()

    def test_n_terms_sums_over_every_order(self):
        from math import comb

        model = FermiVanillaMpfPlusKernelClassifier(n_qubits=4, sectors=[2, 3])
        assert model.n_terms == comb(8, 2) + comb(8, 3)

    def test_a_single_even_order_reproduces_the_default_code_path(self, x_train, y_train):
        # The extended matrix's leading 2n block is the plain covariance matrix, so routing an even order
        # through it must change nothing. This is what makes any difference at odd orders attributable.
        # The encoder is a LazyLinear drawing from torch's global generator, which random_state does not
        # reach, so both models are built from the same seeded state exactly as SingleExperiment does.
        import torch

        torch.manual_seed(0)
        extended = FermiVanillaMpfPlusKernelClassifier(n_qubits=4, sectors=[2], random_state=0)
        extended.fit(x_train, y_train)
        torch.manual_seed(0)
        plain = FermiVanillaMpfPlusKernelClassifier(n_qubits=4, sector_m=2, random_state=0)
        plain.fit(x_train, y_train)
        np.testing.assert_allclose(extended.transform(x_train), plain.transform(x_train), rtol=1e-5, atol=1e-7)

    def test_odd_order_features_are_the_displacement_vector_at_order_one(self, x_train, y_train):
        from matchcake.utils import sector_pfaffian_features

        model = FermiVanillaMpfPlusKernelClassifier(n_qubits=4, sectors=[1], random_state=0)
        model.fit(x_train, y_train)
        x = to_tensor(model.scaler.transform(x_train), dtype=model.R_DTYPE)
        model._q_device.execute_generator(model.ansatz(x), reset=True)
        extended = np.asarray(to_tensor(model._q_device.extended_covariance_matrix).detach().real)
        features = np.asarray(
            to_tensor(sector_pfaffian_features(extended, model.index_sets_per_order[0])).detach().real
        )
        np.testing.assert_allclose(features, extended[..., : model.parity_index, model.parity_index], atol=1e-6)

    def test_multi_sector_kernel_is_symmetric(self, x_train, y_train):
        model = FermiVanillaMpfPlusKernelClassifier(n_qubits=4, sectors=[2, 3])
        model.fit(x_train, y_train)
        kernel_matrix = model.transform(x_train)
        np.testing.assert_allclose(
            kernel_matrix, kernel_matrix.T, atol=ATOL_MATRIX_COMPARISON, rtol=RTOL_MATRIX_COMPARISON
        )

    def test_multi_sector_predicts(self, x_train, y_train):
        model = FermiVanillaMpfPlusKernelClassifier(n_qubits=4, sectors=[2, 3])
        model.fit(x_train, y_train)
        assert model.predict(x_train).shape == (3,)

    def test_odd_order_is_feasible_where_the_even_bound_would_reject_it(self):
        # n=2 leaves 2n-2=2, which blocks order four, but an odd order only needs m <= 2n.
        model = FermiVanillaMpfPlusKernelClassifier(n_qubits=2, sectors=[3])
        model.fit(np.arange(6).reshape(3, 2).astype(np.float32), np.array([0, 1, 0]))
        assert model.n_terms == 4

    def test_an_order_beyond_the_index_count_is_rejected(self):
        model = FermiVanillaMpfPlusKernelClassifier(n_qubits=2, sectors=[5])
        with pytest.raises(ValueError, match="no non-trivial subsets"):
            model.fit(np.arange(6).reshape(3, 2).astype(np.float32), np.array([0, 1, 0]))

    def test_the_vacuum_variant_refuses_sectors(self):
        # Odd orders ride on the displacement vector, which is zero on the vacuum state: silently accepting
        # them there would produce an all-zero feature map and a degenerate Gram matrix rather than an error.
        with pytest.raises(TypeError):
            FermiVanillaMpfKernelClassifier(n_qubits=4, sectors=[2, 3])

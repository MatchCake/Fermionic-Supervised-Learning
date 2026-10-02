from math import comb

import numpy as np
import pytest
from matchcake.utils.torch_utils import to_tensor

from fermiml_framework_classification.models.fermi_angle_mpf_kernel_classifier import (
    FermiAngleMpfKernelClassifier,
)
from tests.configs import ATOL_MATRIX_COMPARISON, RTOL_MATRIX_COMPARISON


class TestFermiAngleMpfKernelClassifier:
    @pytest.fixture
    def model(self):
        return FermiAngleMpfKernelClassifier()

    @pytest.fixture
    def x_train(self):
        # 10 features so n_qubits=10 (even) is compatible with SptmAngleEmbedding
        return np.arange(30).reshape(3, 10).astype(np.float32)

    @pytest.fixture
    def y_train(self):
        return np.array([0, 1, 0]).astype(int)

    def test_init(self, model):
        assert model.MODEL_NAME == "AngleMpf"

    def test_default_rotations(self, model):
        assert model.rotations == FermiAngleMpfKernelClassifier.DEFAULT_ROTATIONS

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

    def test_explicit_n_qubits_skips_default_flag(self):
        model = FermiAngleMpfKernelClassifier(n_qubits=4)
        assert model._n_qubits_is_default_flag is False
        assert model.n_qubits == 4

    def test_fit_preserves_explicit_n_qubits(self, x_train, y_train):
        # SptmAngleEmbedding maps features 1-to-1 to wires, so n_qubits must equal n_features.
        n_features = x_train.shape[-1]
        model = FermiAngleMpfKernelClassifier(n_qubits=n_features)
        model.fit(x_train, y_train)
        assert model.n_qubits == n_features

    def test_covariance_diagonal_not_all_ones(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        kernel_matrix = model.transform(x_train)
        diagonal = np.diag(kernel_matrix)
        assert not np.allclose(diagonal, np.ones_like(diagonal), atol=ATOL_MATRIX_COMPARISON)

    def test_sector_m_default(self, model):
        assert model.sector_m == FermiAngleMpfKernelClassifier.DEFAULT_SECTOR_M

    def test_sector_m_invalid_odd(self):
        with pytest.raises(ValueError, match="sector_m"):
            FermiAngleMpfKernelClassifier(sector_m=3)

    def test_sector_m_invalid_zero(self):
        with pytest.raises(ValueError, match="sector_m"):
            FermiAngleMpfKernelClassifier(sector_m=0)

    def test_sector_m_exceeds_bound_with_explicit_n_qubits(self):
        with pytest.raises(ValueError, match="sector_m"):
            FermiAngleMpfKernelClassifier(n_qubits=2, sector_m=4)

    def test_sector_m_exceeds_bound_detected_at_fit(self, x_train, y_train):
        model = FermiAngleMpfKernelClassifier(sector_m=4)
        with pytest.raises(ValueError, match="sector_m"):
            model.fit(x_train[:, :2], y_train)

    def test_n_terms(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        assert model.n_terms == comb(2 * model.n_qubits, model.sector_m)

    def test_compute_similarities_same_tensor_uses_v0(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        x = to_tensor(model.scaler.transform(x_train), dtype=model.R_DTYPE)
        gram = model.compute_similarities(x, x)
        assert gram.shape == (len(x_train), len(x_train))

    def test_sector_m_setter_valid_value(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        model.sector_m = 2
        assert model.sector_m == 2
        assert "index_sets" not in model.__dict__

    def test_sector_m_setter_invalid_value(self, model):
        with pytest.raises(ValueError, match="sector_m"):
            model.sector_m = 3

    def test_custom_rotations(self, x_train, y_train):
        model = FermiAngleMpfKernelClassifier(rotations="X,Y")
        model.fit(x_train, y_train)
        assert model.rotations == "X,Y"
        predictions = model.predict(x_train)
        assert predictions.shape == (3,)

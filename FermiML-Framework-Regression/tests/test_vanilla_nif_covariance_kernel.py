import numpy as np
import pytest
import torch
from matchcake.ml.kernels.linear_nif_kernel import LinearNIFKernel

from fermiml_framework_regression.nif_covariance_kernel import NIFCovarianceKernel
from fermiml_framework_regression.vanilla_nif_covariance_kernel import VanillaNIFCovarianceKernel


@pytest.mark.parametrize("n_qubits, sector_m, bias", [(n, m, b) for n in [3, 4] for m in [2, 4] for b in [True, False]])
class TestVanillaNIFCovarianceKernel:
    @pytest.fixture(scope="function")
    def kernel_instance(self, n_qubits, sector_m, bias):
        return VanillaNIFCovarianceKernel(n_qubits=n_qubits, sector_m=sector_m, bias=bias)

    @pytest.fixture
    def x_train(self, n_qubits):
        rng = np.random.default_rng(0)
        return rng.uniform(0.0, 0.5, size=(8, n_qubits)).astype(np.float32)

    def test_mro_ansatz_from_linear(self, kernel_instance):
        # ansatz must resolve to LinearNIFKernel (the covariance base leaves it abstract)
        assert type(kernel_instance).ansatz is LinearNIFKernel.ansatz

    def test_mro_machinery_from_covariance(self, kernel_instance):
        assert type(kernel_instance).compute_similarities is NIFCovarianceKernel.compute_similarities
        assert type(kernel_instance).feature_map is NIFCovarianceKernel.feature_map

    def test_fit_then_freeze(self, kernel_instance, x_train):
        # LazyLinear cannot be frozen before it materialises; fit first.
        kernel_instance.fit(x_train)
        kernel_instance.freeze()
        assert kernel_instance.is_fitted_

    def test_symmetric_and_psd(self, kernel_instance, x_train):
        kernel_instance.fit(x_train)
        gram = kernel_instance(x_train)
        torch.testing.assert_close(gram, gram.T)
        eigmin = np.linalg.eigvalsh(gram.detach().cpu().numpy()).min()
        assert eigmin > -1e-5

    def test_diagonal_is_not_one(self, kernel_instance, x_train):
        kernel_instance.fit(x_train)
        gram = kernel_instance(x_train).detach().cpu().numpy()
        assert not np.allclose(np.diag(gram), 1.0)


def test_n_qubits_setter_rebuilds_encoder_and_cache():
    kernel = VanillaNIFCovarianceKernel(n_qubits=3, sector_m=2)
    assert kernel.n_terms == 15  # C(6, 2)
    kernel.n_qubits = 4
    assert kernel.n_qubits == 4
    assert kernel.n_terms == 28  # C(8, 2) -> cache invalidated
    # encoder LazyLinear output width tracks 2n upper-triangular entries
    assert kernel.encoder[1].out_features == kernel.encoder_out_indices[0].size


def test_svm_pipeline_with_alignment_off():
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    X, y = _toy_binary_dataset()
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    kernel = VanillaNIFCovarianceKernel(n_qubits=4, sector_m=2)
    kernel.fit(x_tr)  # materialise LazyLinear
    kernel.freeze()
    k_tr = kernel(x_tr, x_tr).detach().cpu().numpy()
    k_te = kernel(x_te, x_tr).detach().cpu().numpy()
    clf = SVC(kernel="precomputed").fit(k_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(k_te))
    assert acc >= 0.0  # smoke test: pipeline runs end-to-end


def _toy_binary_dataset(n=60, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, np.pi, size=(n, d)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > X[:, 2] + X[:, 3]).astype(int)
    return X, y

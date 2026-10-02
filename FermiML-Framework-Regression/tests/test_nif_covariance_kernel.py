import numpy as np
import pennylane as qml
import pytest
import torch
from matchcake.operations import MAngleEmbedding

from fermiml_framework_regression.angle_nif_covariance_kernel import AngleNIFCovarianceKernel
from fermiml_framework_regression.sklearn_nif_covariance_kernel import SklearnNIFCovarianceKernel

# the shipped concrete angle-embedding kernel exercises the abstract base's machinery
_AngleCovarianceKernel = AngleNIFCovarianceKernel


def _statevector_z_expvals(x, n_qubits, rotations="Y,X"):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circ(k):
        for rot in rotations.split(","):
            MAngleEmbedding(x, wires=range(n_qubits), rotations=rot)
        return qml.expval(qml.Z(k))

    return np.array([float(circ(k)) for k in range(n_qubits)])


@pytest.mark.parametrize("n_qubits, sector_m", [(n, m) for n in [4, 6] for m in [2, 4]])
class TestNIFCovarianceKernel:
    @pytest.fixture(scope="function")
    def kernel_instance(self, n_qubits, sector_m):
        return _AngleCovarianceKernel(n_qubits=n_qubits, sector_m=sector_m).freeze()

    @pytest.fixture
    def x_train(self, n_qubits):
        rng = np.random.default_rng(0)
        return rng.uniform(0.0, np.pi, size=(8, n_qubits)).astype(np.float32)

    def test_feature_dimension(self, kernel_instance, n_qubits, sector_m):
        from math import comb

        assert kernel_instance.n_terms == comb(2 * n_qubits, sector_m)
        assert kernel_instance.index_sets.shape == (comb(2 * n_qubits, sector_m), sector_m)

    def test_fit(self, kernel_instance, x_train):
        kernel_instance.fit(x_train)
        assert kernel_instance.x_train_ is x_train
        assert kernel_instance.is_fitted_

    def test_symmetric_kernel(self, kernel_instance, x_train):
        gram = kernel_instance(x_train)
        torch.testing.assert_close(gram, gram.T)

    def test_positive_semidefinite(self, kernel_instance, x_train):
        gram = kernel_instance(x_train).detach().cpu().numpy()
        eigmin = np.linalg.eigvalsh(gram).min()
        assert eigmin > -1e-5, f"Gram matrix not PSD: min eig {eigmin}"

    def test_diagonal_is_not_one(self, kernel_instance, x_train):
        # Unlike a fidelity kernel, the diagonal is (c/2^n)||v_m(x)||^2 != 1.
        gram = kernel_instance(x_train).detach().cpu().numpy()
        assert not np.allclose(np.diag(gram), 1.0)

    def test_diag_matches_gram_diagonal(self, kernel_instance, x_train):
        adapter = SklearnNIFCovarianceKernel(kernel_instance)
        gram = kernel_instance(x_train).detach().cpu().numpy()
        np.testing.assert_allclose(adapter.diag(x_train), np.diag(gram), atol=1e-5)


@pytest.mark.parametrize("n_qubits", [4, 6])
def test_feature_map_reproduces_z_expvals(n_qubits):
    """For m=2, the signed Pfaffian features recover <Z_k> exactly.

    With Z_k = -i c_{2k} c_{2k+1} and the Wick phase i^{-1}, one has
    <Z_k> = -Pf(Lambda|{2k,2k+1}). This is the core correctness check that
    ties the feature map to physical expectation values.
    """
    rng = np.random.default_rng(1)
    x = rng.uniform(0.0, np.pi, size=(1, n_qubits)).astype(np.float32)
    kernel = _AngleCovarianceKernel(n_qubits=n_qubits, sector_m=2).freeze()

    kernel._q_device.execute_generator(kernel.ansatz(torch.as_tensor(x)), reset=True)
    from matchcake.utils import sector_pfaffian_features

    idx2 = np.array([[2 * k, 2 * k + 1] for k in range(n_qubits)])
    zk_from_pfaffian = -np.real(
        np.asarray(sector_pfaffian_features(kernel._q_device.covariance_matrix, idx2).detach().cpu())
    ).ravel()
    zk_statevector = _statevector_z_expvals(x[0], n_qubits)
    np.testing.assert_allclose(zk_from_pfaffian, zk_statevector, atol=1e-5)


def test_observable_norm_scales_gram():
    rng = np.random.default_rng(2)
    x = rng.uniform(0.0, np.pi, size=(6, 4)).astype(np.float32)
    k1 = _AngleCovarianceKernel(n_qubits=4, sector_m=2, observable_norm=1.0).freeze()
    k2 = _AngleCovarianceKernel(n_qubits=4, sector_m=2, observable_norm=3.0).freeze()
    g1 = k1(x).detach().cpu().numpy()
    g2 = k2(x).detach().cpu().numpy()
    np.testing.assert_allclose(g2, 3.0 * g1, atol=1e-5)


def test_sector_m_validation():
    with pytest.raises(ValueError):
        _AngleCovarianceKernel(n_qubits=4, sector_m=3)  # odd
    with pytest.raises(ValueError):
        _AngleCovarianceKernel(n_qubits=2, sector_m=4)  # exceeds 2n-2


def test_sector_m_setter_invalidates_cache():
    kernel = _AngleCovarianceKernel(n_qubits=4, sector_m=2)
    assert kernel.n_terms == 28  # C(8, 2)
    kernel.sector_m = 4
    assert kernel.n_terms == 70  # C(8, 4)


def test_n_qubits_setter_invalidates_cache():
    kernel = _AngleCovarianceKernel(n_qubits=3, sector_m=2)
    assert kernel.n_terms == 15  # C(6, 2)
    kernel.n_qubits = 4
    assert kernel.n_qubits == 4
    assert kernel.n_terms == 28  # C(8, 2)


def test_svm_pipeline():
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    rng = np.random.default_rng(0)
    n_qubits = 6
    X = rng.uniform(0.0, np.pi, size=(120, n_qubits)).astype(np.float32)
    gt = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(gt)
    def magn(x):
        MAngleEmbedding(x, wires=range(n_qubits), rotations="Y")
        MAngleEmbedding(x, wires=range(n_qubits), rotations="X")
        return qml.expval(qml.sum(*[qml.Z(k) for k in range(n_qubits)]))

    y = np.array([float(magn(x)) for x in X])
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, train_size=60, random_state=0)
    thr = np.median(y_tr)
    y_tr_b, y_te_b = (y_tr > thr).astype(int), (y_te > thr).astype(int)

    kernel = _AngleCovarianceKernel(n_qubits=n_qubits, sector_m=2).freeze()
    k_tr = kernel(x_tr, x_tr).detach().cpu().numpy()
    k_te = kernel(x_te, x_tr).detach().cpu().numpy()
    clf = SVC(kernel="precomputed", C=10.0).fit(k_tr, y_tr_b)
    # the magnetisation target lives in B_2, so the m=2 kernel should classify well
    assert accuracy_score(y_te_b, clf.predict(k_te)) > 0.8


def test_gp_adapter_pipeline():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(0)
    n_qubits = 6
    X = rng.uniform(0.0, np.pi, size=(120, n_qubits)).astype(np.float32)
    gt = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(gt)
    def magn(x):
        MAngleEmbedding(x, wires=range(n_qubits), rotations="Y")
        MAngleEmbedding(x, wires=range(n_qubits), rotations="X")
        return qml.expval(qml.sum(*[qml.Z(k) for k in range(n_qubits)]))

    y = np.array([float(magn(x)) for x in X])
    x_tr, x_te, y_tr, y_te = train_test_split(X, y, train_size=40, random_state=1)

    kernel = _AngleCovarianceKernel(n_qubits=n_qubits, sector_m=2).freeze()
    ym, ys = y_tr.mean(), y_tr.std()
    gp = GaussianProcessRegressor(kernel=SklearnNIFCovarianceKernel(kernel), alpha=1e-4, optimizer=None)
    gp.fit(x_tr, (y_tr - ym) / ys)
    pred = gp.predict(x_te) * ys + ym
    assert r2_score(y_te, pred) > 0.9

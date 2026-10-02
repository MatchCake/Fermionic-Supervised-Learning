import numpy as np
import pytest
from sklearn.base import clone

from fermiml_framework_regression.models.angle_mpf_plus_gp_regressor import AngleMpfPlusGPRegressor
from fermiml_framework_regression.models.angle_nif_gp_regressor import AngleNIFGPRegressor
from fermiml_framework_regression.models.vanilla_mpf_plus_gp_regressor import VanillaMpfPlusGPRegressor
from fermiml_framework_regression.models.vanilla_nif_gp_regressor import VanillaNIFGPRegressor
from fermiml_framework_regression.sklearn_nif_covariance_kernel import SklearnNIFCovarianceKernel

N_FEATURES = 8

MODEL_FACTORIES = {
    "AngleNIFGP": (AngleNIFGPRegressor, {"sector_m": 2}),
    "VanillaNIFGP": (VanillaNIFGPRegressor, {"sector_m": 2}),
    "AngleMpfPlus": (AngleMpfPlusGPRegressor, {"sectors": [1, 2, 3]}),
    "VanillaMpfPlus": (VanillaMpfPlusGPRegressor, {"sectors": [1, 2, 3]}),
}


@pytest.fixture
def training_data():
    rng = np.random.RandomState(0)
    x = rng.uniform(0.0, 1.0, size=(48, N_FEATURES)).astype(np.float32)
    y = np.sin(x.sum(axis=1)).astype(np.float32)
    return x, y


@pytest.mark.parametrize("model_name", sorted(MODEL_FACTORIES))
class TestConfiguredKernelSurvivesTheGPClone:
    """The kernel the GP evaluates must be the kernel that was configured.

    ``GaussianProcessRegressor.fit`` opens with ``self.kernel_ = clone(self.kernel)``, and
    :func:`sklearn.base.clone` rebuilds an estimator from ``get_params()``. These kernels are
    ``BaseEstimator`` subclasses through matchcake's ``Kernel``, and they pass most of their
    arguments through ``**kwargs``, which sklearn's constructor-signature introspection cannot
    see -- ``get_params()`` reported as little as ``{'sectors'}``. The reconstruction therefore
    reverted the rest to class defaults, and ``NIFKernel.DEFAULT_N_QUBITS`` is 12, so a model
    configured for an 8-feature dataset silently fitted a 12-qubit kernel with a feature map
    several times the configured size.

    Nothing detected it: ``SptmAngleEmbedding`` only rejects a qubit count that is too *small*,
    fit and predict were consistent with each other because both go through the clone, and
    ``resolved_n_qubits`` in the results tree reads ``nif_kernel_`` -- the object that was built
    and then thrown away -- so the recorded diagnostics described a model that never ran.
    """

    def test_the_gp_uses_the_kernel_that_was_built(self, model_name, training_data):
        x, y = training_data
        model_cls, kwargs = MODEL_FACTORIES[model_name]
        model = model_cls(n_qubits=-1, alpha=1e-4, **kwargs).fit(x, y)
        assert model.gp_.kernel_.kernel is model.nif_kernel_

    def test_qubit_count_is_the_data_dimension_not_the_class_default(self, model_name, training_data):
        """The regression itself: ``n_qubits=-1`` must resolve to 8 here, never to the default 12."""
        x, y = training_data
        model_cls, kwargs = MODEL_FACTORIES[model_name]
        model = model_cls(n_qubits=-1, alpha=1e-4, **kwargs).fit(x, y)
        assert model.nif_kernel_.n_qubits == N_FEATURES
        assert model.gp_.kernel_.kernel.n_qubits == N_FEATURES
        # n_terms follows from n_qubits, so it is the quantity the capacity argument in the paper
        # is stated in and the one a silent qubit-count change corrupts.
        assert model.gp_.kernel_.kernel.n_terms == model.nif_kernel_.n_terms

    def test_a_non_default_batch_size_reaches_the_kernel(self, model_name, training_data):
        """``gram_batch_size`` travels the same path and was reverted to 10000 by the same clone."""
        x, y = training_data
        model_cls, kwargs = MODEL_FACTORIES[model_name]
        model = model_cls(n_qubits=-1, alpha=1e-4, gram_batch_size=16, **kwargs).fit(x, y)
        assert model.gp_.kernel_.kernel.gram_batch_size == 16


def test_cloning_the_wrapper_preserves_the_wrapped_kernel(training_data):
    """Guard the mechanism directly, without a GP in the way."""
    x, y = training_data
    model = AngleMpfPlusGPRegressor(n_qubits=-1, sectors=[1, 2, 3], alpha=1e-4).fit(x, y)
    wrapper = SklearnNIFCovarianceKernel(model.nif_kernel_, scale=1.0)
    cloned = clone(wrapper)
    assert cloned.kernel is model.nif_kernel_
    assert cloned.kernel.n_qubits == N_FEATURES

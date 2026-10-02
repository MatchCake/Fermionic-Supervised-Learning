import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor

from fermiml_framework_regression.models.angle_nif_gp_regressor import AngleNIFGPRegressor
from fermiml_framework_regression.models.nif_gp_regressor import NIFGPRegressor
from fermiml_framework_regression.models.vanilla_nif_gp_regressor import VanillaNIFGPRegressor

MODEL_FACTORIES = {
    "AngleNIFGP": AngleNIFGPRegressor,
    "VanillaNIFGP": VanillaNIFGPRegressor,
}


@pytest.fixture
def training_data():
    rng = np.random.RandomState(0)
    x = rng.uniform(0.0, np.pi, size=(24, 4)).astype(np.float32)
    y = np.sin(x).sum(axis=1).astype(np.float32)
    return x, y


@pytest.mark.parametrize("model_name", sorted(MODEL_FACTORIES))
class TestJitterEscalation:
    """A Gram matrix a rounding error away from indefinite must cost jitter, not the whole job.

    The covariance kernel's feature map has :math:`\\binom{2n}{m}` dimensions, so a Gram matrix
    over more training points than that is singular by construction and only ``alpha`` keeps it
    factorizable. The kernel is evaluated in float32, which fills the null space with noise of
    the same order as the smallest alpha in the sweep, so whether the Cholesky survives comes
    down to the BLAS and the thread count: the two jobs that failed the cluster sweep at
    ``alpha=1e-6`` both fit locally. Escalating the jitter is what makes that difference stop
    being the difference between a result and a lost job.
    """

    def test_requested_alpha_is_used_when_it_works(self, model_name, training_data):
        x, y = training_data
        model = MODEL_FACTORIES[model_name](n_qubits=-1, sector_m=2, alpha=1e-4)
        model.fit(x, y)
        assert model.n_jitter_retries_ == 0
        assert model.effective_alpha_ == pytest.approx(1e-4)

    def test_escalates_and_records_when_cholesky_fails(self, model_name, training_data, monkeypatch):
        x, y = training_data
        model = MODEL_FACTORIES[model_name](n_qubits=-1, sector_m=2, alpha=1e-6)
        # The real failure depends on the BLAS, so it is forced here instead: the first two
        # alphas raise exactly as sklearn does, and the third is left to fit for real.
        original_fit = GaussianProcessRegressor.fit
        rejected = []

        def failing_fit(self, x_train, y_train):
            # Compared against a threshold between the second and third alpha rather than
            # against 1e-4: repeated multiplication by 10 lands on 9.999...e-05, which is
            # below 1e-4 and would reject the attempt this test means to let through.
            if self.alpha < 5e-5:
                rejected.append(self.alpha)
                raise np.linalg.LinAlgError("2-th leading minor of the array is not positive definite")
            return original_fit(self, x_train, y_train)

        monkeypatch.setattr(GaussianProcessRegressor, "fit", failing_fit)
        with pytest.warns(RuntimeWarning, match="not positive definite"):
            model.fit(x, y)

        assert rejected == [pytest.approx(1e-6), pytest.approx(1e-5)]
        assert model.n_jitter_retries_ == 2
        assert model.effective_alpha_ == pytest.approx(1e-4)
        assert np.all(np.isfinite(model.predict(x)))

    def test_reraises_when_every_escalation_fails(self, model_name, training_data, monkeypatch):
        x, y = training_data
        model = MODEL_FACTORIES[model_name](n_qubits=-1, sector_m=2, alpha=1e-6)
        attempts = []

        def always_failing_fit(self, x_train, y_train):
            attempts.append(self.alpha)
            raise np.linalg.LinAlgError("2-th leading minor of the array is not positive definite")

        monkeypatch.setattr(GaussianProcessRegressor, "fit", always_failing_fit)
        with pytest.raises(np.linalg.LinAlgError):
            model.fit(x, y)

        # Escalation is bounded: a genuinely broken kernel must still surface as a failure
        # rather than be regularized into meaninglessness.
        assert len(attempts) == NIFGPRegressor.JITTER_MAX_RETRIES + 1
        assert not hasattr(model, "effective_alpha_")

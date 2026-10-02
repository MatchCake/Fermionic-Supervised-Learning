import numpy as np
import pytest

from fermiml_framework_regression.models.angle_nif_gp_regressor import AngleNIFGPRegressor
from fermiml_framework_regression.models.nif_gp_regressor import NIFGPRegressor
from fermiml_framework_regression.models.vanilla_nif_gp_regressor import VanillaNIFGPRegressor
from fermiml_framework_regression.preprocessing import X_SCALINGS

MODEL_FACTORIES = {
    "AngleNIFGP": AngleNIFGPRegressor,
    "VanillaNIFGP": VanillaNIFGPRegressor,
}


@pytest.fixture
def wide_range_data():
    """Inputs whose raw units leave one period of the embedding, as California Housing's do."""
    rng = np.random.RandomState(0)
    x = rng.uniform(0.0, 500.0, size=(32, 4)).astype(np.float32)
    y = np.sin(x[:, 0] / 500.0 * 3.0).astype(np.float32)
    return x, y


@pytest.fixture
def heavy_tailed_data():
    """One column with a tail long enough that min-max leaves the bulk unresolvable.

    This is the shape California Housing's ``AveOccup`` has: a single extreme value sets the
    min-max denominator and squashes 99% of the column into a fraction of a percent of the
    embedding's period.
    """
    rng = np.random.RandomState(0)
    x = rng.uniform(0.0, 1.0, size=(200, 4)).astype(np.float32)
    x[:, 1] = rng.exponential(1.0, size=200).astype(np.float32)
    x[0, 1] = 500.0
    y = (x[:, 1] > np.median(x[:, 1])).astype(np.float32)
    return x, y


@pytest.mark.parametrize("model_name", sorted(MODEL_FACTORIES))
class TestInputScaling:
    """The transform that decides what the embedding actually sees.

    The matchgate embedding is periodic in its inputs, so a feature carried in raw units can
    wrap it many times over and become noise. Bounding it is necessary but not sufficient: a
    heavy-tailed column min-maxed into ``[0, 1]`` sits inside one period while occupying a
    fraction of a percent of it, which the circuit cannot resolve either. Hence a named
    transform rather than a window, and hence ``"quantile"``.
    """

    def test_scales_into_the_requested_window(self, model_name, wide_range_data):
        x, y = wide_range_data
        model = MODEL_FACTORIES[model_name](n_qubits=-1, sector_m=2, alpha=1e-4, x_scaling="minmax01")
        model.fit(x, y)
        assert model.x_scaler_ is not None
        scaled = model._scale_x(x)
        assert scaled.min() == pytest.approx(0.0, abs=1e-6)
        assert scaled.max() == pytest.approx(1.0, abs=1e-6)

    def test_default_passes_inputs_through_untouched(self, model_name, wide_range_data):
        """Off by default, so enabling it is a recorded choice rather than a silent one."""
        x, y = wide_range_data
        model = MODEL_FACTORIES[model_name](n_qubits=-1, sector_m=2, alpha=1e-4)
        model.fit(x, y)
        assert model.x_scaler_ is None
        np.testing.assert_array_equal(model._scale_x(x), x)

    @pytest.mark.parametrize("x_scaling", X_SCALINGS)
    def test_every_named_transform_fits_and_predicts(self, model_name, x_scaling, wide_range_data):
        """Each name in the swept set has to survive a full fit/predict, not just construct."""
        x, y = wide_range_data
        model = MODEL_FACTORIES[model_name](n_qubits=-1, sector_m=2, alpha=1e-4, x_scaling=x_scaling)
        model.fit(x, y)
        assert np.all(np.isfinite(model.predict(x)))

    def test_an_unknown_transform_is_rejected_at_construction(self, model_name):
        """Caught where it is written, not several hours into a cluster job."""
        with pytest.raises(ValueError, match="Unknown x_scaling"):
            MODEL_FACTORIES[model_name](n_qubits=-1, sector_m=2, x_scaling="minmax")

    def test_quantile_spreads_a_heavy_tail_that_minmax_squashes(self, model_name, heavy_tailed_data):
        """The reason "quantile" is in the set at all.

        Both transforms land the column inside ``[0, 1]``, so a bounds check cannot tell them
        apart. What separates them is how much of that interval the bulk of the data occupies:
        min-max leaves it in a sliver the embedding cannot resolve, the rank transform spreads
        it across the whole window.
        """
        x, y = heavy_tailed_data
        spans = {}
        for x_scaling in ("minmax01", "quantile"):
            model = MODEL_FACTORIES[model_name](n_qubits=-1, sector_m=2, alpha=1e-4, x_scaling=x_scaling)
            model.fit(x, y)
            column = model._scale_x(x)[:, 1]
            assert 0.0 <= column.min() and column.max() <= 1.0 + 1e-6, x_scaling
            # The middle 90%, so the span measures where the bulk sits rather than being pulled
            # back up to 1.0 by the single extreme value that caused the problem.
            spans[x_scaling] = float(np.percentile(column, 95) - np.percentile(column, 5))
        assert spans["minmax01"] < 0.05, spans
        assert spans["quantile"] > 0.8, spans

    def test_scaler_is_fitted_on_training_data_only(self, model_name, wide_range_data):
        """A test point beyond the training range must fall outside the window, not be clipped.

        If the scaler were refitted on whatever it is asked to transform, every split would be
        rescaled to its own extremes and test information would leak into the transform.
        """
        x, y = wide_range_data
        model = MODEL_FACTORIES[model_name](n_qubits=-1, sector_m=2, alpha=1e-4, x_scaling="minmax01")
        model.fit(x, y)
        beyond = np.full((1, x.shape[1]), 2.0 * x.max(), dtype=np.float32)
        assert model._scale_x(beyond).max() > 1.0

    def test_predict_applies_the_same_transform_as_fit(self, model_name, wide_range_data):
        """Predicting on the training inputs must reproduce the in-sample fit.

        A scaler applied in fit but forgotten in predict would leave the model evaluating the
        kernel at raw coordinates, which is silent: the numbers stay finite and only the
        accuracy collapses.
        """
        x, y = wide_range_data
        model = MODEL_FACTORIES[model_name](n_qubits=-1, sector_m=2, alpha=1e-6, x_scaling="minmax01")
        model.fit(x, y)
        predictions = model.predict(x)
        assert np.all(np.isfinite(predictions))
        # A GP with a tiny jitter interpolates its own training data closely; if predict saw
        # unscaled inputs the kernel row would be unrelated to the fitted one and this fails.
        assert np.corrcoef(predictions, y)[0, 1] > 0.9

    def test_scaling_rescues_a_wide_range_target(self, model_name, wide_range_data):
        """The measured effect: bounding the inputs beats leaving them raw on wide-range data."""
        x, y = wide_range_data
        scores = {}
        for x_scaling in ("none", "minmax01"):
            model = MODEL_FACTORIES[model_name](n_qubits=-1, sector_m=2, alpha=1e-6, x_scaling=x_scaling)
            model.fit(x, y)
            scores[x_scaling] = float(np.corrcoef(model.predict(x), y)[0, 1])
        assert scores["minmax01"] >= scores["none"]


def test_scaling_is_off_by_default():
    """Pinned deliberately: turning it on by default breaks the magnetisation calibration.

    That dataset's inputs are already angles in [0, pi] and its target is built to sit exactly
    in the matched kernel's RKHS; rescaling them takes the recovery test from 0.99 to 0.59. The
    transform is therefore swept in data/configs/config.yaml rather than defaulted on, so each
    cell gets the choice validation prefers.
    """
    assert NIFGPRegressor.DEFAULT_X_SCALING == "none"

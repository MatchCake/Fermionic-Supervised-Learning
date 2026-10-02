import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

from fermiml_framework_regression.models import (
    AngleMpfPlusGPRegressor,
    AngleNIFGPRegressor,
    ClassicalGPRegressor,
    ClassicalSVR,
    VanillaMpfPlusGPRegressor,
    VanillaNIFGPRegressor,
    VanillaNIFKernelRegressor,
)
from fermiml_framework_regression.preprocessing import (
    MAX_QUANTILE_KNOTS,
    X_SCALINGS,
    TargetRangeClipMixin,
    make_x_scaler,
)

N_FEATURES = 4

# Every model in the sweep, built small enough to fit in a unit test. The clip and the input
# transform are applied by all of them or the benchmark is not a comparison, so both are
# checked against the whole set rather than the covariance GPs that motivated them.
MODEL_FACTORIES = {
    "ClassicalGP": lambda **kwargs: ClassicalGPRegressor(**kwargs),
    "ClassicalSVR": lambda **kwargs: ClassicalSVR(**kwargs),
    "VanillaSVR": lambda **kwargs: VanillaNIFKernelRegressor(n_qubits=-1, **kwargs),
    "AngleNIFGP": lambda **kwargs: AngleNIFGPRegressor(n_qubits=-1, sector_m=2, **kwargs),
    "VanillaNIFGP": lambda **kwargs: VanillaNIFGPRegressor(n_qubits=-1, sector_m=2, **kwargs),
    "AngleMpfPlus": lambda **kwargs: AngleMpfPlusGPRegressor(n_qubits=-1, sectors=[1, 2], **kwargs),
    "VanillaMpfPlus": lambda **kwargs: VanillaMpfPlusGPRegressor(n_qubits=-1, sectors=[1, 2], **kwargs),
}


@pytest.fixture
def training_data():
    rng = np.random.RandomState(0)
    x = rng.uniform(0.0, 1.0, size=(40, N_FEATURES)).astype(np.float32)
    y = (3.0 * x[:, 0] + x[:, 1]).astype(np.float32)
    return x, y


class TestMakeXScaler:
    """The transform factory behind the swept ``x_scaling`` axis."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("minmax01", MinMaxScaler),
            ("minmax11", MinMaxScaler),
            ("standard", StandardScaler),
            ("quantile", QuantileTransformer),
        ],
    )
    def test_each_name_builds_its_transformer(self, name, expected):
        assert isinstance(make_x_scaler(name, n_samples=100), expected)

    @pytest.mark.parametrize("name", [None, "none"])
    def test_the_pass_through_names_build_nothing(self, name):
        assert make_x_scaler(name, n_samples=100) is None

    def test_an_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown x_scaling"):
            make_x_scaler("minmax", n_samples=100)

    def test_every_swept_name_is_constructible(self):
        """X_SCALINGS is what the config sweeps, so no member may be a name the factory rejects."""
        for name in X_SCALINGS:
            make_x_scaler(name, n_samples=100)

    @pytest.mark.parametrize(("n_samples", "expected"), [(10, 10), (5000, MAX_QUANTILE_KNOTS)])
    def test_quantile_knots_are_clamped_to_the_sample_count(self, n_samples, expected):
        """sklearn warns and silently clamps when n_quantiles exceeds the sample count.

        Small cells in this grid go down to 160 training points, so the unclamped default of
        1000 would emit a warning on most of the sweep.
        """
        assert make_x_scaler("quantile", n_samples=n_samples).n_quantiles == expected

    def test_the_minmax_windows_differ(self):
        """minmax01 and minmax11 are both in the swept set, so they must not be the same object."""
        assert make_x_scaler("minmax01", 100).feature_range == (0.0, 1.0)
        assert make_x_scaler("minmax11", 100).feature_range == (-1.0, 1.0)

    def test_the_quantile_transform_is_seeded(self):
        """Above ``subsample`` points the CDF is estimated from a random draw.

        The default subsample is 10000 and the largest California Housing cell trains on 13209
        points, so an unseeded transform is genuinely non-reproducible there: two fits of the
        same data disagreed by 4.9e-3. A hash-addressed results tree cannot tolerate that, and
        nothing else in the suite pins it.
        """
        rng = np.random.RandomState(0)
        x = rng.lognormal(size=(12000, 3))
        first = make_x_scaler("quantile", len(x)).fit_transform(x)
        np.random.seed(999)
        second = make_x_scaler("quantile", len(x)).fit_transform(x)
        np.testing.assert_array_equal(first, second)


class TestTargetRangeClipMath:
    """The property that makes clipping safe to apply unconditionally.

    For any target inside ``[low, high]``, moving a prediction into that interval cannot
    increase the squared error: either the prediction was already inside and nothing changes,
    or it was outside and the clip moves it toward every point of the interval at once. That is
    why this is applied to every model rather than being swept, and it is what the measured
    result reflects: over the 5040 completed California Housing runs, clipping improved 4336 and
    made none worse.
    """

    def test_clipping_never_increases_squared_error_for_an_in_range_target(self):
        rng = np.random.RandomState(0)
        low, high = -1.0, 2.0
        targets = rng.uniform(low, high, size=5000)
        predictions = rng.uniform(-50.0, 50.0, size=5000)
        clipped = np.clip(predictions, low, high)
        assert np.all((clipped - targets) ** 2 <= (predictions - targets) ** 2 + 1e-12)

    def test_clipping_does_hurt_an_out_of_range_target(self):
        """The other half of the guarantee, pinned so nobody reads the first test as universal.

        A model that extrapolates correctly past the training range is dragged back and
        penalised. This regime is absent from the shipped grid -- the worst cell, California
        Housing at pool 500, puts 16 of 4128 test targets outside the training range and leaves
        an oracle at 0.999961 -- but a new dataset with an unbounded target would reopen it.
        """
        low, high = 0.0, 10.0
        targets = np.linspace(10.0, 11.5, 50)
        clipped = np.clip(targets, low, high)
        r2 = 1.0 - ((targets - clipped) ** 2).sum() / ((targets - targets.mean()) ** 2).sum()
        assert r2 < 0.0, r2

    def test_a_prediction_already_inside_is_untouched(self):
        class Model(TargetRangeClipMixin):
            pass

        model = Model()
        model._fit_target_range(np.array([0.0, 10.0]))
        inside = np.array([0.0, 3.5, 10.0])
        np.testing.assert_array_equal(model._clip_to_target_range(inside), inside)

    def test_the_range_comes_from_the_training_targets(self):
        class Model(TargetRangeClipMixin):
            pass

        model = Model()
        model._fit_target_range(np.array([2.0, -1.0, 7.0, 0.0]))
        assert (model.y_train_min_, model.y_train_max_) == (-1.0, 7.0)


@pytest.mark.parametrize("model_name", sorted(MODEL_FACTORIES))
class TestEveryModelClipsAndScales:
    """Applied uniformly, or the comparison this package exists to make is rigged.

    Until 0.0.6 the input transform was hardcoded per model and only the covariance GPs swept
    anything. Adding a transform to the matchgate kernels alone would have tuned one side of the
    benchmark against a pinned baseline, so both the transform and the clip go to every model.
    """

    def test_the_training_target_range_is_recorded(self, model_name, training_data):
        x, y = training_data
        model = MODEL_FACTORIES[model_name]().fit(x, y)
        assert model.y_train_min_ == pytest.approx(float(y.min()))
        assert model.y_train_max_ == pytest.approx(float(y.max()))

    def test_predictions_stay_inside_the_training_target_range(self, model_name, training_data):
        """Probed well outside the training inputs, which is where the overshoot happens."""
        x, y = training_data
        model = MODEL_FACTORIES[model_name]().fit(x, y)
        far = np.random.RandomState(1).uniform(-30.0, 30.0, size=(50, N_FEATURES)).astype(np.float32)
        predictions = model.predict(far)
        assert predictions.min() >= float(y.min()) - 1e-6
        assert predictions.max() <= float(y.max()) + 1e-6

    def test_the_range_ignores_targets_never_shown_to_fit(self, model_name, training_data):
        """The bound is training-only, so nothing leaks in from validation or test.

        The held-out half is given a target far above anything ``fit`` sees, so a bound computed
        over the whole array rather than the training split would be unmistakable.
        """
        x, _ = training_data
        y = np.linspace(0.0, 1.0, len(x)).astype(np.float32)
        y[len(x) // 2 :] += 100.0
        model = MODEL_FACTORIES[model_name]().fit(x[: len(x) // 2], y[: len(x) // 2])
        assert model.y_train_max_ == pytest.approx(float(y[: len(x) // 2].max()))
        assert model.y_train_max_ < float(y.max())

    @pytest.mark.parametrize("x_scaling", X_SCALINGS)
    def test_every_transform_survives_fit_and_predict(self, model_name, x_scaling, training_data):
        x, y = training_data
        model = MODEL_FACTORIES[model_name](x_scaling=x_scaling).fit(x, y)
        assert np.all(np.isfinite(model.predict(x)))

    def test_an_unknown_transform_is_rejected_at_construction(self, model_name):
        with pytest.raises(ValueError, match="Unknown x_scaling"):
            MODEL_FACTORIES[model_name](x_scaling="not-a-transform")

    def test_the_default_transform_is_a_swept_value(self, model_name):
        """Each model's historical hardcoded choice has to remain reachable from the sweep."""
        model = MODEL_FACTORIES[model_name]()
        assert model.x_scaling in X_SCALINGS


@pytest.mark.parametrize("x_scaling", X_SCALINGS)
class TestTheFidelityKernelIsSelfConsistent:
    """The fidelity kernel's reference set must be in the same units as its queries.

    ``Kernel.fit`` stores its argument as ``x_train_`` and ``transform(x)`` evaluates
    ``K(x, x_train_)``, while ``predict`` scales its query. Handing the parent the raw inputs
    therefore fitted the SVR on ``K(scaled, raw)``, silently: the numbers stay finite and only
    the accuracy suffers. It went unnoticed because the only assertion covering the training
    Gram checked its diagonal, which ``GramMatrix`` fills with 1.0 unconditionally and which
    cannot fail whatever the inputs are.
    """

    def test_the_reference_set_is_the_scaled_training_data(self, x_scaling, training_data):
        x, y = training_data
        model = VanillaNIFKernelRegressor(n_qubits=-1, x_scaling=x_scaling).fit(x, y)
        np.testing.assert_allclose(np.asarray(model.x_train_), model._scale_x(x), rtol=1e-6, atol=1e-8)

    def test_scaling_inside_the_model_matches_scaling_by_hand(self, x_scaling, training_data):
        """The end-to-end consequence, independent of how the reference set is stored.

        Pre-scaling the inputs and fitting with ``"none"`` has to agree with letting the model
        do the same transform itself. Under the bug these differed by 9.03 in prediction units.
        """
        x, y = training_data
        internal = VanillaNIFKernelRegressor(n_qubits=-1, x_scaling=x_scaling).fit(x, y)
        prescaled = internal._scale_x(x).astype(np.float32)
        external = VanillaNIFKernelRegressor(n_qubits=-1, x_scaling="none").fit(prescaled, y)
        np.testing.assert_allclose(internal.predict(x), external.predict(prescaled), rtol=1e-5, atol=1e-6)


class TestClipDoesNotHideAGoodModel:
    """A clip that fired on a well-behaved model would silently flatten it.

    On Friedman and magnetisation the replay measured it as an exact no-op, so the guarantee
    worth pinning is that an in-range model is returned untouched rather than merely improved.
    """

    def test_predict_returns_exactly_the_clipped_raw_prediction(self, training_data):
        """The clip is the only thing between the GP posterior mean and what predict returns.

        Pinned as an identity rather than an inequality so that a future change which rescales
        or otherwise perturbs the prediction on its way out cannot hide behind the bound.
        """
        x, y = training_data
        model = VanillaNIFGPRegressor(n_qubits=-1, sector_m=2, alpha=1e-4, x_scaling="minmax01")
        model.fit(x, y)
        raw = model.gp_.predict(model._scale_x(x))
        raw = model.y_scaler_.inverse_transform(raw.reshape(-1, 1)).ravel()
        expected = np.clip(raw, float(y.min()), float(y.max()))
        np.testing.assert_allclose(model.predict(x), expected, rtol=1e-6, atol=1e-8)

    def test_an_in_range_prediction_passes_through_untouched(self, training_data):
        """Where the model does not overshoot, the clip must be a bit-exact no-op.

        This is the property the replay measured on Friedman and magnetisation, where the
        clipped and unclipped scores were identical to four decimals.
        """
        x, y = training_data
        model = VanillaNIFGPRegressor(n_qubits=-1, sector_m=2, alpha=1e-4, x_scaling="minmax01")
        model.fit(x, y)
        raw = model.gp_.predict(model._scale_x(x))
        raw = model.y_scaler_.inverse_transform(raw.reshape(-1, 1)).ravel()
        inside = (raw >= float(y.min())) & (raw <= float(y.max()))
        assert inside.any(), "fixture no longer exercises the in-range case"
        np.testing.assert_array_equal(model.predict(x)[inside], raw[inside])

    def test_predictive_std_is_not_clipped(self, training_data):
        """A spread is not a location; bounding it by the target range would be meaningless."""
        x, y = training_data
        model = VanillaNIFGPRegressor(n_qubits=-1, sector_m=2, alpha=1e-4).fit(x, y)
        assert np.all(model.predict_std(x) >= 0.0)
        assert np.all(model.predict_std(x) < float(y.max()))

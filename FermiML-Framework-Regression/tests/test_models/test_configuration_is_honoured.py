import inspect
import warnings
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import yaml
from sklearn.base import clone

from fermiml_framework_regression.experiments.single_experiment import SingleExperiment
from fermiml_framework_regression.models.angle_mpf_plus_gp_regressor import AngleMpfPlusGPRegressor
from fermiml_framework_regression.models.angle_nif_gp_regressor import AngleNIFGPRegressor
from fermiml_framework_regression.models.classical_gp_regressor import ClassicalGPRegressor
from fermiml_framework_regression.models.classical_svr import ClassicalSVR
from fermiml_framework_regression.models.vanilla_mpf_plus_gp_regressor import VanillaMpfPlusGPRegressor
from fermiml_framework_regression.models.vanilla_nif_gp_regressor import VanillaNIFGPRegressor
from fermiml_framework_regression.models.vanilla_nif_kernel_regressor import VanillaNIFKernelRegressor
from tests.configs import CONFIG_PATH

N_FEATURES = 4
N_TRAIN = 20
N_TEST = 8
MIN_EFFECT = 1e-6
MODEL_CLASSES: Dict[str, Any] = {
    cls.MODEL_NAME: cls  # type: ignore[attr-defined]
    for cls in SingleExperiment.ALLOWED_MODEL_CLS
}
BEHAVIOURAL: Dict[str, Dict[str, Tuple[Any, Any]]] = {
    "ClassicalGP": {
        "kernel_length_scale": (0.1, 10.0),
        "alpha": (1e-8, 1e-1),
        "x_scaling": ("standard", "quantile"),
    },
    "ClassicalSVR": {
        "svr_c": (0.1, 100.0),
        "svr_epsilon": (0.01, 1.0),
        "gamma": ("scale", "auto"),
        "x_scaling": ("minmax11", "quantile"),
    },
    "VanillaSVR": {
        "n_qubits": (N_FEATURES, N_FEATURES + 2),
        "svr_c": (0.1, 100.0),
        "svr_epsilon": (0.01, 1.0),
        "bias": (True, False),
        "encoder_activation": ("Identity", "Tanh"),
        "random_state": (0, 12345),
        "x_scaling": ("minmax11", "quantile"),
    },
    "AngleNIFGP": {
        "n_qubits": (N_FEATURES, N_FEATURES + 2),
        "sector_m": (2, 4),
        "rotations": ("Y,X", "X,Y"),
        "alpha": (1e-6, 1e-1),
        "scale": (1.0, 7.0),
        "x_scaling": ("none", "quantile"),
    },
    "VanillaNIFGP": {
        "n_qubits": (N_FEATURES, N_FEATURES + 2),
        "sector_m": (2, 4),
        "bias": (True, False),
        "encoder_activation": ("Identity", "Tanh"),
        "alpha": (1e-6, 1e-1),
        "scale": (1.0, 7.0),
        "x_scaling": ("none", "quantile"),
        "random_state": (0, 12345),
    },
    "AngleMpfPlus": {
        "n_qubits": (N_FEATURES, N_FEATURES + 2),
        "sectors": ([1, 2, 3], [1, 2, 3, 4]),
        "rotations": ("Y,X", "X,Y"),
        "alpha": (1e-6, 1e-1),
        "scale": (1.0, 7.0),
        "x_scaling": ("none", "quantile"),
    },
    "VanillaMpfPlus": {
        "n_qubits": (N_FEATURES, N_FEATURES + 2),
        "sectors": ([1, 2, 3], [1, 2, 3, 4]),
        "bias": (True, False),
        "encoder_activation": ("Identity", "Tanh"),
        "alpha": (1e-6, 1e-1),
        "scale": (1.0, 7.0),
        "x_scaling": ("none", "quantile"),
        "random_state": (0, 12345),
    },
}
STRUCTURAL: Dict[str, Dict[str, Tuple[str, Any]]] = {
    "ClassicalGP": {"n_restarts_optimizer": ("gp_", 7)},
    "VanillaSVR": {"gram_batch_size": ("self", 8)},
    "AngleNIFGP": {"gram_batch_size": ("kernel", 8), "random_state": ("kernel", 12345)},
    "VanillaNIFGP": {"gram_batch_size": ("kernel", 8), "random_state": ("kernel", 12345)},
    "AngleMpfPlus": {"gram_batch_size": ("kernel", 8), "random_state": ("kernel", 12345)},
    "VanillaMpfPlus": {"gram_batch_size": ("kernel", 8), "random_state": ("kernel", 12345)},
}
ALIGNMENT_PARAMS = {
    "alignment",
    "alignment_iterations",
    "alignment_learning_rate",
    "alignment_early_stopping_patience",
    "alignment_early_stopping_threshold",
}


def _data():
    rng = np.random.RandomState(0)
    x = rng.uniform(0.0, 1.0, size=(N_TRAIN, N_FEATURES)).astype(np.float32)
    y = np.sin(x.sum(axis=1)).astype(np.float32)
    x_test = rng.uniform(0.0, 1.0, size=(N_TEST, N_FEATURES)).astype(np.float32)
    return x, y, x_test


@pytest.fixture
def data():
    return _data()


def _predict_with(model_name: str, **kwargs) -> np.ndarray:
    """Fit a model with ``kwargs`` and return its predictions on a fixed test set."""
    x, y, x_test = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MODEL_CLASSES[model_name](**kwargs).fit(x, y)
        return np.asarray(model.predict(x_test), dtype=float)


def _kernel_used_at_predict_time(model):
    """The kernel object the fitted model actually evaluates, whatever kind of model it is.

    VanillaSVR *is* its own kernel; the GP regressors hold theirs inside the fitted
    ``GaussianProcessRegressor``, which is the copy the cloning bug used to replace.
    """
    gp = getattr(model, "gp_", None)
    fitted = getattr(gp, "kernel_", None)
    return getattr(fitted, "kernel", model) if fitted is not None else model


BEHAVIOURAL_CASES = [(model, name) for model, params in BEHAVIOURAL.items() for name in params]
STRUCTURAL_CASES = [(model, name) for model, params in STRUCTURAL.items() for name in params]


@pytest.mark.parametrize(("model_name", "parameter"), BEHAVIOURAL_CASES, ids=lambda v: str(v))
class TestParametersChangeTheModel:
    """Configuring a parameter differently must produce a different model.

    The strong form of the contract, and the one that would have caught the cloning bug without
    knowing anything about it: a value that never reaches the computation produces identical
    predictions for two different settings, no matter how faithfully it is stored elsewhere.
    """

    def test_two_values_give_different_predictions(self, model_name, parameter):
        value_a, value_b = BEHAVIOURAL[model_name][parameter]
        predictions_a = _predict_with(model_name, **{parameter: value_a})
        predictions_b = _predict_with(model_name, **{parameter: value_b})
        difference = float(np.abs(predictions_a - predictions_b).max())
        assert difference > MIN_EFFECT, (
            f"{model_name}.{parameter}: {value_a!r} and {value_b!r} give the same predictions "
            f"(max |difference| = {difference:.3e}), so the setting is not reaching the model."
        )

    def test_the_same_value_is_reproducible(self, model_name, parameter):
        """Guards the comparison above: a difference only means something if repeats agree."""
        value_a, _ = BEHAVIOURAL[model_name][parameter]
        first = _predict_with(model_name, **{parameter: value_a})
        second = _predict_with(model_name, **{parameter: value_a})
        np.testing.assert_allclose(first, second, rtol=0, atol=0)


@pytest.mark.parametrize(("model_name", "parameter"), STRUCTURAL_CASES, ids=lambda v: str(v))
class TestBehaviourNeutralParametersStillReach:
    """Parameters that are neutral by design still have to arrive.

    ``gram_batch_size`` chunks a computation whose parts are independent, and ``random_state``
    seeds an encoder that the fixed-ansatz Angle models do not have, so neither shows up in the
    predictions. That makes them exactly the parameters a silent reset can hide in -- the
    cloning bug reverted ``gram_batch_size`` from the configured value to 10000 in the same
    breath as ``n_qubits``, and only the latter was visible in the numbers.
    """

    def test_the_value_is_present_on_the_object_that_computes(self, model_name, parameter, data):
        x, y, _ = data
        where, value = STRUCTURAL[model_name][parameter]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MODEL_CLASSES[model_name](**{parameter: value}).fit(x, y)
        if where == "kernel":
            target = _kernel_used_at_predict_time(model)
        elif where == "self":
            target = model
        else:
            target = getattr(model, where)
        assert getattr(target, parameter) == value


class TestEveryConstructorParameterIsCovered:
    """No parameter may be added without deciding how it is verified.

    Coverage is asserted against the constructor signatures rather than maintained by hand, so a
    new hyperparameter fails this test until it is classified as behavioural, structural, or an
    alignment knob. That is the property that keeps this suite honest as the models grow.
    """

    @pytest.mark.parametrize("model_name", sorted(MODEL_CLASSES))
    def test_no_parameter_is_unverified(self, model_name):
        signature = inspect.signature(MODEL_CLASSES[model_name].__init__)
        declared = {name for name in signature.parameters if name != "self"}
        covered = set(BEHAVIOURAL.get(model_name, {})) | set(STRUCTURAL.get(model_name, {})) | ALIGNMENT_PARAMS
        assert not declared - covered, (
            f"{model_name} has parameters with no verification: {sorted(declared - covered)}. "
            f"Add each to BEHAVIOURAL if it changes predictions, or to STRUCTURAL if it cannot."
        )

    @pytest.mark.parametrize("model_name", sorted(MODEL_CLASSES))
    def test_no_verification_targets_a_parameter_that_no_longer_exists(self, model_name):
        signature = inspect.signature(MODEL_CLASSES[model_name].__init__)
        declared = {name for name in signature.parameters if name != "self"}
        claimed = set(BEHAVIOURAL.get(model_name, {})) | set(STRUCTURAL.get(model_name, {}))
        assert not claimed - declared, f"{model_name}: {sorted(claimed - declared)} are no longer constructor arguments"


@pytest.mark.parametrize("model_name", sorted(MODEL_CLASSES))
class TestSklearnCloneOfTheRegressor:
    """``clone`` of the regressor must be a faithful copy.

    This is the same mechanism that broke the kernels, one level up. It holds today because no
    regressor constructor takes ``**kwargs`` -- sklearn's ``get_params`` reads the signature, so
    an argument it cannot see is an argument it silently drops. Pinned here because adding a
    ``**kwargs`` to any of these constructors would reintroduce exactly the original bug for
    anything that clones a regressor, which is every sklearn cross-validation utility.
    """

    def test_clone_preserves_every_parameter(self, model_name):
        signature = inspect.signature(MODEL_CLASSES[model_name].__init__)
        declared = {name for name in signature.parameters if name != "self"}
        model = MODEL_CLASSES[model_name]()
        assert declared <= set(model.get_params()), (
            f"{model_name}.get_params() does not report {sorted(declared - set(model.get_params()))}; "
            f"sklearn.clone would reset them to defaults."
        )

    def test_a_fitted_clone_predicts_identically(self, model_name, data):
        x, y, x_test = data
        settings = {name: values[0] for name, values in BEHAVIOURAL.get(model_name, {}).items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original = MODEL_CLASSES[model_name](**settings).fit(x, y)
            copy = clone(MODEL_CLASSES[model_name](**settings)).fit(x, y)
            np.testing.assert_allclose(original.predict(x_test), copy.predict(x_test), rtol=0, atol=0)


class TestTheShippedGridIsHonoured:
    """Every value the sweep actually runs must reach the model.

    The tables above are chosen by hand, so they can drift from what is swept. This reads
    ``data/configs/config.yaml`` and checks the real grid: for each hyperparameter with more than
    one value, the first and last must give different predictions. A hyperparameter that is added
    to the sweep but silently ignored costs a whole cluster run to discover otherwise -- which is
    the bill that was just paid.
    """

    @staticmethod
    def _grid():
        config = yaml.safe_load(CONFIG_PATH.read_text())
        cases = []
        for entry in config["models"]:
            for name, values in (entry.get("hyperparameters") or {}).items():
                if isinstance(values, list) and len(values) > 1:
                    cases.append((entry["name"], name, values[0], values[-1]))
        return cases

    def test_the_grid_sweeps_something(self):
        assert self._grid(), "no model in the shipped config sweeps more than one value"

    def test_every_swept_hyperparameter_changes_the_model(self):
        failures = []
        for model_name, parameter, first, last in self._grid():
            # n_qubits is swept only as the auto-detect sentinel, which resolves to the data's
            # feature count; a two-value sweep of it does not occur and is not meaningful here.
            if parameter == "n_qubits":
                continue
            predictions_first = _predict_with(model_name, **{parameter: first})
            predictions_last = _predict_with(model_name, **{parameter: last})
            difference = float(np.abs(predictions_first - predictions_last).max())
            if difference <= MIN_EFFECT:
                failures.append(f"{model_name}.{parameter}: {first!r} vs {last!r} -> max |d| = {difference:.3e}")
        assert not failures, "swept hyperparameters with no effect on the model:\n  " + "\n  ".join(failures)


class TestKernelAlignmentIsNotAvailable:
    """Pins a capability the notes claim and the code does not have.

    ``alignment`` is off throughout the shipped grid, and the paper's open items describe it as
    something the covariance kernel supports but that has not been tried. Turning it on raises
    ``RuntimeError: expected scalar type Double but found Float`` from inside the alignment
    optimiser, for every kernel model. Recorded as a strict xfail rather than a
    :func:`pytest.raises`, so that fixing it upstream turns this green and forces the claim --
    and the alignment knobs' exemption from coverage above -- to be revisited.
    """

    @pytest.mark.xfail(strict=True, reason="alignment optimiser fails on a float32 kernel; see docs/paper open items")
    @pytest.mark.parametrize(
        "model_cls",
        [AngleNIFGPRegressor, VanillaNIFGPRegressor, AngleMpfPlusGPRegressor, VanillaMpfPlusGPRegressor,
         VanillaNIFKernelRegressor],
    )  # fmt: skip
    def test_alignment_can_be_enabled(self, model_cls, data):
        x, y, _ = data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_cls(alignment=True, alignment_iterations=2).fit(x, y)


def test_every_model_in_the_config_is_constructible():
    """The registry and the shipped grid must name the same models."""
    config = yaml.safe_load(CONFIG_PATH.read_text())
    configured = {entry["name"] for entry in config["models"]}
    assert configured <= set(MODEL_CLASSES), f"config names unknown models: {sorted(configured - set(MODEL_CLASSES))}"
    assert {ClassicalGPRegressor, ClassicalSVR}.issubset(set(MODEL_CLASSES.values()))

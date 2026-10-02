import numpy as np
import pytest

from fermiml_framework_regression.datasets.friedman import FriedmanDataset
from fermiml_framework_regression.experiments.configs.dataset_config import DatasetConfig
from fermiml_framework_regression.experiments.configs.model_config import ModelConfig
from fermiml_framework_regression.experiments.configs.single_experiment_config import SingleExperimentConfig
from fermiml_framework_regression.experiments.single_experiment import SingleExperiment
from fermiml_framework_regression.models.angle_mpf_plus_gp_regressor import AngleMpfPlusGPRegressor
from fermiml_framework_regression.models.angle_nif_gp_regressor import AngleNIFGPRegressor
from fermiml_framework_regression.models.classical_gp_regressor import ClassicalGPRegressor
from fermiml_framework_regression.models.classical_svr import ClassicalSVR
from fermiml_framework_regression.models.vanilla_mpf_plus_gp_regressor import VanillaMpfPlusGPRegressor
from fermiml_framework_regression.models.vanilla_nif_gp_regressor import VanillaNIFGPRegressor
from fermiml_framework_regression.models.vanilla_nif_kernel_regressor import VanillaNIFKernelRegressor
from fermiml_framework_regression.preprocessing import X_SCALINGS


MODEL_FACTORIES = {
    "ClassicalGP": lambda: ClassicalGPRegressor(kernel_length_scale=1.0),
    "ClassicalSVR": lambda: ClassicalSVR(svr_c=10.0, svr_epsilon=0.1),
    "VanillaSVR": lambda: VanillaNIFKernelRegressor(n_qubits=-1, svr_c=0.1, svr_epsilon=0.1),
    "AngleNIFGP": lambda: AngleNIFGPRegressor(n_qubits=-1, sector_m=2, alpha=1e-4),
    "VanillaNIFGP": lambda: VanillaNIFGPRegressor(n_qubits=-1, sector_m=2, alpha=1e-4),
    "AngleMpfPlus": lambda: AngleMpfPlusGPRegressor(n_qubits=-1, sectors=(1, 2, 3), alpha=1e-4),
    "VanillaMpfPlus": lambda: VanillaMpfPlusGPRegressor(n_qubits=-1, sectors=(1, 2, 3), alpha=1e-4),
}


@pytest.fixture(scope="module")
def training_data():
    dataset = FriedmanDataset(n_samples=48, n_test_samples=16, cv_splits=4, val_size=0.25)
    dataset.prepare_data()
    return dataset


@pytest.mark.parametrize("model_name", sorted(MODEL_FACTORIES))
class TestModelDeterminism:
    """Two fits of one configuration must agree exactly.

    Results are addressed by a hash of their configuration, so a model that is not reproducible
    silently breaks that contract: a requeued or re-run job overwrites its own cached answer
    with a different number, and the published figure cannot be regenerated. The trainable
    ansatz models draw their encoder weights from torch's global RNG, which ``random_state``
    does not reach, so this is a real failure mode rather than a hypothetical one.
    """

    def test_repeated_fits_agree(self, model_name, training_data):
        factory = MODEL_FACTORIES[model_name]
        first = factory()
        first.fit(training_data.x_train, training_data.y_train)
        first_predictions = first.predict(training_data.x_test)

        second = factory()
        second.fit(training_data.x_train, training_data.y_train)
        second_predictions = second.predict(training_data.x_test)

        np.testing.assert_allclose(
            first_predictions,
            second_predictions,
            rtol=0,
            atol=0,
            err_msg=f"{model_name} is not reproducible across fits of the same configuration",
        )

    @pytest.mark.parametrize("x_scaling", X_SCALINGS)
    def test_repeated_fits_agree_under_every_input_transform(self, model_name, x_scaling, training_data):
        """The transform is part of the configuration, so it is part of the contract.

        ``quantile`` is the one that can break it: above its ``subsample`` threshold the
        transform estimates the CDF from a random draw, so without a fixed ``random_state`` the
        same configuration gives different numbers on the largest California Housing cell. The
        global RNGs are disturbed between the two fits so a model leaning on them cannot pass.
        """
        factory = MODEL_FACTORIES[model_name]
        first = factory()
        first.set_params(x_scaling=x_scaling)
        first.fit(training_data.x_train, training_data.y_train)
        first_predictions = first.predict(training_data.x_test)

        np.random.seed(4321)

        second = factory()
        second.set_params(x_scaling=x_scaling)
        second.fit(training_data.x_train, training_data.y_train)

        np.testing.assert_allclose(
            first_predictions,
            second.predict(training_data.x_test),
            rtol=0,
            atol=0,
            err_msg=f"{model_name} is not reproducible at x_scaling={x_scaling!r}",
        )

    def test_predict_is_stable(self, model_name, training_data):
        model = MODEL_FACTORIES[model_name]()
        model.fit(training_data.x_train, training_data.y_train)
        np.testing.assert_allclose(
            model.predict(training_data.x_test),
            model.predict(training_data.x_test),
            rtol=0,
            atol=0,
            err_msg=f"{model_name} gives different predictions on repeated calls",
        )


class TestRunnerSeeding:
    def test_seed_is_derived_from_the_config_hash(self):
        config = SingleExperimentConfig(
            dataset=DatasetConfig(dataset_name="friedman", split_id=0, cv_splits=20),
            model=ModelConfig(name="ClassicalGP", hyperparameters={"kernel_length_scale": 1.0}),
        )
        assert SingleExperiment.seed_everything(_StubExperiment(config)) == int(config.hash[:8], 16)

    def test_different_configs_get_different_seeds(self):
        first = SingleExperimentConfig(
            dataset=DatasetConfig(dataset_name="friedman", split_id=0, cv_splits=20),
            model=ModelConfig(name="ClassicalGP", hyperparameters={"kernel_length_scale": 1.0}),
        )
        second = SingleExperimentConfig(
            dataset=DatasetConfig(dataset_name="friedman", split_id=1, cv_splits=20),
            model=ModelConfig(name="ClassicalGP", hyperparameters={"kernel_length_scale": 1.0}),
        )
        assert first.hash != second.hash
        assert SingleExperiment.seed_everything(_StubExperiment(first)) != SingleExperiment.seed_everything(
            _StubExperiment(second)
        )


class _StubExperiment:
    """The two attributes ``seed_everything`` touches, without building an OutputFolder.

    Constructing a real :class:`SingleExperiment` writes into the results tree, which a unit
    test must not do.
    """

    def __init__(self, config: SingleExperimentConfig) -> None:
        self.config = config

    def log_func(self, message: str) -> None:
        pass

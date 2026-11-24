import numpy as np
import pytest
import torch
from matchcake.utils.torch_utils import to_numpy
from tqdm import tqdm

from fermiml_framework_classification.models.fermi_angle_mpf_kernel_classifier import (
    FermiAngleMpfKernelClassifier,
)
from fermiml_framework_classification.models.fermi_iqp_kernel_classifier import (
    FermiIQPKernelClassifier,
)
from fermiml_framework_classification.models.fermi_vanilla_kernel_classifier import (
    FermiVanillaKernelClassifier,
)
from fermiml_framework_classification.models.fermi_vanilla_mpf_kernel_classifier import (
    FermiVanillaMpfKernelClassifier,
)
from fermiml_framework_classification.models.fermi_vanilla_mpf_plus_kernel_classifier import (
    FermiVanillaMpfPlusKernelClassifier,
)
from tests.configs import ATOL_SINGLE_PRECISION_COMPARISON, TEST_SEED

ALL_CLASSIFIERS = [
    FermiIQPKernelClassifier,
    FermiVanillaKernelClassifier,
    FermiVanillaMpfKernelClassifier,
    FermiVanillaMpfPlusKernelClassifier,
    FermiAngleMpfKernelClassifier,
]
FIDELITY_CLASSIFIERS = [FermiIQPKernelClassifier, FermiVanillaKernelClassifier]


class TestNIFKernelClassifier:
    @classmethod
    def _entry_by_entry(cls, model, x: np.ndarray) -> np.ndarray:
        """Evaluate the fidelity kernel one pair at a time, without going through ``GramMatrix``."""
        scaled = torch.as_tensor(model.scaler.transform(x), dtype=model.R_DTYPE)
        sptm0, sptm1 = model._x_to_sptm(scaled), model.sptms_train
        bar = tqdm(disable=True)
        return np.array(
            [
                [
                    float(
                        torch.real(
                            torch.as_tensor(model._compute_similarities_func(np.array([[row, col]]), sptm0, sptm1, bar))
                        ).reshape(-1)[0]
                    )
                    for col in range(sptm1.shape[0])
                ]
                for row in range(sptm0.shape[0])
            ]
        )

    @pytest.fixture
    def x_train(self):
        # Deliberately away from [-1, 1], where scaling twice would be close to scaling once.
        return np.random.default_rng(TEST_SEED).uniform(-3.0, 5.0, (12, 4)).astype(np.float32)

    @pytest.fixture
    def y_train(self):
        return np.array([0, 1] * 6).astype(int)

    @pytest.fixture
    def x_test(self):
        return np.random.default_rng(TEST_SEED + 1).uniform(-3.0, 5.0, (5, 4)).astype(np.float32)

    @staticmethod
    def _fit(model_cls, x_train, y_train):
        torch.manual_seed(TEST_SEED)
        return model_cls(random_state=TEST_SEED).fit(x_train, y_train)

    @pytest.mark.parametrize("model_cls", ALL_CLASSIFIERS)
    def test_predict_applies_the_scaler_once(self, model_cls, x_train, y_train, x_test, monkeypatch):
        model = self._fit(model_cls, x_train, y_train)
        seen = []
        original = model.scaler.transform
        monkeypatch.setattr(model.scaler, "transform", lambda x: (seen.append(np.array(x, copy=True)), original(x))[1])
        model.predict(x_test)
        scaled = original(x_test)
        assert not any(call.shape == scaled.shape and np.allclose(call, scaled) for call in seen)

    @pytest.mark.parametrize("model_cls", ALL_CLASSIFIERS)
    def test_predict_matches_the_svm_on_the_kernel_to_the_training_set(self, model_cls, x_train, y_train, x_test):
        model = self._fit(model_cls, x_train, y_train)
        np.testing.assert_array_equal(model.predict(x_test), model.svm.predict(to_numpy(model.transform(x_test))))
        np.testing.assert_allclose(
            model.predict_proba(x_test), model.svm.predict_proba(to_numpy(model.transform(x_test)))
        )

    @pytest.mark.parametrize("model_cls", ALL_CLASSIFIERS)
    def test_predict_on_training_points_uses_the_training_kernel(self, model_cls, x_train, y_train):
        model = self._fit(model_cls, x_train, y_train)
        np.testing.assert_array_equal(model.predict(x_train), model.svm.predict(to_numpy(model.transform(x_train))))

    @pytest.mark.parametrize("model_cls", FIDELITY_CLASSIFIERS)
    def test_rectangular_kernel_matches_entry_by_entry_evaluation(self, model_cls, x_train, y_train, x_test):
        model = self._fit(model_cls, x_train, y_train)
        np.testing.assert_allclose(
            to_numpy(model.transform(x_test)),
            self._entry_by_entry(model, x_test),
            atol=ATOL_SINGLE_PRECISION_COMPARISON,
        )

    @pytest.mark.parametrize("model_cls", FIDELITY_CLASSIFIERS)
    def test_square_kernel_between_different_points_is_not_mirrored(self, model_cls, x_train, y_train):
        model = self._fit(model_cls, x_train, y_train)
        others = np.random.default_rng(TEST_SEED + 2).uniform(-3.0, 5.0, x_train.shape).astype(np.float32)
        kernel = to_numpy(model.transform(others))
        np.testing.assert_allclose(kernel, self._entry_by_entry(model, others), atol=ATOL_SINGLE_PRECISION_COMPARISON)
        assert not np.allclose(kernel, kernel.T, atol=ATOL_SINGLE_PRECISION_COMPARISON)

    @pytest.mark.parametrize("model_cls", FIDELITY_CLASSIFIERS)
    def test_more_test_points_than_training_points(self, model_cls, x_train, y_train):
        model = self._fit(model_cls, x_train, y_train)
        others = np.random.default_rng(TEST_SEED + 3).uniform(-3.0, 5.0, (15, 4)).astype(np.float32)
        np.testing.assert_allclose(
            to_numpy(model.transform(others)),
            self._entry_by_entry(model, others),
            atol=ATOL_SINGLE_PRECISION_COMPARISON,
        )

    @pytest.mark.parametrize("model_cls", FIDELITY_CLASSIFIERS)
    def test_training_kernel_is_symmetric_with_a_unit_diagonal(self, model_cls, x_train, y_train):
        model = self._fit(model_cls, x_train, y_train)
        kernel = to_numpy(model.transform(x_train))
        np.testing.assert_allclose(kernel, kernel.T, atol=ATOL_SINGLE_PRECISION_COMPARISON)
        np.testing.assert_allclose(np.diag(kernel), 1.0, atol=ATOL_SINGLE_PRECISION_COMPARISON)
        np.testing.assert_allclose(kernel, self._entry_by_entry(model, x_train), atol=ATOL_SINGLE_PRECISION_COMPARISON)

    @pytest.mark.parametrize("model_cls", FIDELITY_CLASSIFIERS)
    def test_refit_does_not_keep_the_previous_training_set(self, model_cls, x_train, y_train, x_test):
        model = self._fit(model_cls, x_train, y_train)
        model.transform(x_test)
        others = np.random.default_rng(TEST_SEED + 4).uniform(-3.0, 5.0, x_train.shape).astype(np.float32)
        torch.manual_seed(TEST_SEED)
        model.fit(others, y_train)
        np.testing.assert_allclose(
            to_numpy(model.transform(x_test)),
            self._entry_by_entry(model, x_test),
            atol=ATOL_SINGLE_PRECISION_COMPARISON,
        )
        scaled = torch.as_tensor(model.scaler.transform(others), dtype=model.R_DTYPE)
        np.testing.assert_allclose(
            to_numpy(model.sptms_train), to_numpy(model._x_to_sptm(scaled)), atol=ATOL_SINGLE_PRECISION_COMPARISON
        )

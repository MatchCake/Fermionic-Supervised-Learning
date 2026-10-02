import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from fermiml_framework_classification.models.fermi_kitchen_sink_kernel_classifier import (
    FermiKitchenSinkKernelClassifier,
)


class TestFermiKitchenSinkKernelClassifier:
    N_FEATURES = 4
    N_SAMPLES = 6
    N_EPISODES = 2

    @pytest.fixture
    def model(self):
        return FermiKitchenSinkKernelClassifier(n_episodes=self.N_EPISODES, n_qfeatures=2, random_state=0)

    @pytest.fixture
    def x_train(self):
        return np.random.default_rng(0).standard_normal((self.N_SAMPLES, self.N_FEATURES)).astype(np.float32)

    @pytest.fixture
    def y_train(self):
        return np.array([0, 1] * (self.N_SAMPLES // 2)).astype(int)

    @pytest.fixture
    def fitted_model(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        return model

    # --- init ---

    def test_model_name(self, model):
        assert model.MODEL_NAME == "fQKS"

    def test_default_n_episodes(self):
        assert FermiKitchenSinkKernelClassifier().n_episodes == FermiKitchenSinkKernelClassifier.DEFAULT_N_EPISODES

    def test_default_n_qfeatures(self):
        assert FermiKitchenSinkKernelClassifier().n_qfeatures == FermiKitchenSinkKernelClassifier.DEFAULT_N_QFEATURES

    def test_default_var(self):
        assert FermiKitchenSinkKernelClassifier().var == FermiKitchenSinkKernelClassifier.DEFAULT_VAR

    def test_weights_none_before_fit(self, model):
        assert model.weights_ is None

    def test_bias_none_before_fit(self, model):
        assert model.bias_ is None

    def test_cls_head_is_logistic_regression(self, model):
        assert isinstance(model.cls_head, LogisticRegression)

    # --- n_qubits from n_qfeatures ---

    def test_n_qubits_full(self, x_train, y_train):
        m = FermiKitchenSinkKernelClassifier(n_episodes=self.N_EPISODES, n_qfeatures="full", random_state=0)
        m.fit(x_train, y_train)
        expected = self.N_FEATURES + (self.N_FEATURES % 2)
        assert m.n_qubits == expected

    def test_n_qubits_half(self, x_train, y_train):
        m = FermiKitchenSinkKernelClassifier(n_episodes=self.N_EPISODES, n_qfeatures="half", random_state=0)
        m.fit(x_train, y_train)
        n = int(np.ceil(self.N_FEATURES / 2))
        expected = n + (n % 2)
        assert m.n_qubits == expected

    def test_n_qubits_int(self, x_train, y_train):
        m = FermiKitchenSinkKernelClassifier(n_episodes=self.N_EPISODES, n_qfeatures=2, random_state=0)
        m.fit(x_train, y_train)
        assert m.n_qubits == 2

    def test_n_qubits_even_after_fit(self, x_train, y_train):
        m = FermiKitchenSinkKernelClassifier(n_episodes=self.N_EPISODES, n_qfeatures=3, random_state=0)
        m.fit(x_train, y_train)
        assert m.n_qubits % 2 == 0

    # --- fit ---

    def test_fit_returns_self(self, model, x_train, y_train):
        assert model.fit(x_train, y_train) is model

    def test_fit_weights_type(self, fitted_model):
        assert isinstance(fitted_model.weights_, torch.nn.Parameter)

    def test_fit_weights_shape(self, fitted_model):
        assert fitted_model.weights_.shape == (self.N_EPISODES, fitted_model.n_qubits, self.N_FEATURES)

    def test_fit_bias_type(self, fitted_model):
        assert isinstance(fitted_model.bias_, torch.nn.Parameter)

    def test_fit_bias_shape(self, fitted_model):
        assert fitted_model.bias_.shape == (self.N_EPISODES, fitted_model.n_qubits)

    def test_bias_range(self, fitted_model):
        bias = fitted_model.bias_.detach().numpy()
        assert np.all(bias >= 0)
        assert np.all(bias <= 2 * np.pi)

    def test_weights_scale_with_var(self, x_train, y_train):
        m1 = FermiKitchenSinkKernelClassifier(n_episodes=self.N_EPISODES, n_qfeatures=2, var=1.0, random_state=0)
        m4 = FermiKitchenSinkKernelClassifier(n_episodes=self.N_EPISODES, n_qfeatures=2, var=4.0, random_state=0)
        m1.fit(x_train, y_train)
        m4.fit(x_train, y_train)
        std1 = m1.weights_.detach().numpy().std()
        std4 = m4.weights_.detach().numpy().std()
        assert std4 == pytest.approx(std1 * 2.0, rel=0.3)

    def test_cls_head_fitted_after_fit(self, fitted_model):
        check_is_fitted(fitted_model.cls_head)

    def test_different_random_states_give_different_weights(self, x_train, y_train):
        m0 = FermiKitchenSinkKernelClassifier(n_episodes=self.N_EPISODES, n_qfeatures=2, random_state=0)
        m1 = FermiKitchenSinkKernelClassifier(n_episodes=self.N_EPISODES, n_qfeatures=2, random_state=1)
        m0.fit(x_train, y_train)
        m1.fit(x_train, y_train)
        assert not torch.allclose(m0.weights_, m1.weights_)

    # --- predict ---

    def test_predict_returns_ndarray(self, fitted_model, x_train):
        assert isinstance(fitted_model.predict(x_train), np.ndarray)

    def test_predict_shape(self, fitted_model, x_train):
        assert fitted_model.predict(x_train).shape == (self.N_SAMPLES,)

    def test_predict_labels_in_classes(self, fitted_model, x_train):
        result = fitted_model.predict(x_train)
        assert set(np.unique(result)).issubset(set(fitted_model.classes_))

    # --- predict_proba ---

    def test_predict_proba_returns_ndarray(self, fitted_model, x_train):
        assert isinstance(fitted_model.predict_proba(x_train), np.ndarray)

    def test_predict_proba_shape(self, fitted_model, x_train):
        assert fitted_model.predict_proba(x_train).shape == (self.N_SAMPLES, 2)

    def test_predict_proba_sums_to_one(self, fitted_model, x_train):
        result = fitted_model.predict_proba(x_train)
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_non_negative(self, fitted_model, x_train):
        assert np.all(fitted_model.predict_proba(x_train) >= 0)

    # --- reproducibility ---

    def test_same_seed_gives_identical_features(self, x_train, y_train):
        features = []
        for _ in range(2):
            np.random.seed(0)
            model = FermiKitchenSinkKernelClassifier(n_episodes=2, n_qfeatures=2, random_state=0)
            model.fit(x_train, y_train)
            features.append(np.asarray(model.sptms_train, dtype=float).copy())
        np.testing.assert_allclose(features[0], features[1], atol=0, rtol=0)

    def test_episodes_are_not_collapsed(self, x_train, y_train):
        np.random.seed(0)
        model = FermiKitchenSinkKernelClassifier(n_episodes=8, n_qfeatures=2, random_state=0)
        model.fit(x_train, y_train)
        features = np.asarray(model.sptms_train, dtype=float).reshape(len(x_train), 8, model.n_qubits)
        blocks = {features[:, index, :].tobytes() for index in range(8)}
        assert len(blocks) > 1

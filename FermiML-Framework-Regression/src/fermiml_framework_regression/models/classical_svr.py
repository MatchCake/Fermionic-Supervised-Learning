from typing import Union

from numpy._typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ..preprocessing import X_SCALINGS, InputScalingMixin, TargetRangeClipMixin


class ClassicalSVR(InputScalingMixin, TargetRangeClipMixin, BaseEstimator, RegressorMixin):
    """Classical support-vector regressor with an RBF kernel.

    A thin wrapper around :class:`~sklearn.svm.SVR` that includes an input transform and target
    standardization so the epsilon and C parameters have consistent interpretations across
    datasets.

    ``x_scaling`` defaults to ``"minmax11"``, which is what this model hardcoded before the
    transform became a swept axis. See :class:`ClassicalGPRegressor` for why it is swept rather
    than fixed.

    :param svr_c: Regularisation parameter for the SVR.
    :param svr_epsilon: Epsilon tube half-width in standardized target units.
    :param gamma: Kernel coefficient for the RBF. ``"scale"`` uses
        :math:`1 / (n_{\\text{features}} \\cdot \\text{Var}(X))`.
    :param x_scaling: Input transform, one of
        :data:`~fermiml_framework_regression.preprocessing.X_SCALINGS`.
    """

    MODEL_NAME = "ClassicalSVR"
    DEFAULT_SVR_C = 1.0
    DEFAULT_SVR_EPSILON = 0.1
    DEFAULT_GAMMA = "scale"
    DEFAULT_X_SCALING = "minmax11"

    def __init__(
        self,
        *,
        svr_c: float = DEFAULT_SVR_C,
        svr_epsilon: float = DEFAULT_SVR_EPSILON,
        gamma: Union[str, float] = DEFAULT_GAMMA,
        x_scaling: str = DEFAULT_X_SCALING,
    ):
        self.svr_c = svr_c
        self.svr_epsilon = svr_epsilon
        self.gamma = gamma
        if x_scaling is not None and x_scaling not in X_SCALINGS:
            raise ValueError(f"Unknown x_scaling {x_scaling!r}; expected one of {X_SCALINGS} or None.")
        self.x_scaling = x_scaling

    def fit(self, x_train: NDArray, y_train: NDArray) -> "ClassicalSVR":
        """Fit the SVR on training data.

        :param x_train: Training inputs of shape ``(n_samples, n_features)``.
        :param y_train: Training targets of shape ``(n_samples,)``.
        :return: The fitted regressor.
        """
        x_scaled = self._fit_x_scaler(x_train)
        self._fit_target_range(y_train)
        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y_train.reshape(-1, 1)).ravel()
        self.svr_ = SVR(C=self.svr_c, epsilon=self.svr_epsilon, gamma=self.gamma, kernel="rbf")
        self.svr_.fit(x_scaled, y_scaled)
        return self

    def predict(self, x: NDArray) -> NDArray:
        """Predict regression targets.

        :param x: Inputs of shape ``(n_samples, n_features)``.
        :return: Predicted targets of shape ``(n_samples,)``, clipped to the training target
            range.
        """
        y_scaled = self.svr_.predict(self._scale_x(x))
        return self._clip_to_target_range(self.y_scaler_.inverse_transform(y_scaled.reshape(-1, 1)).ravel())

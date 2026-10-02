import numpy as np
from numpy._typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

from ..preprocessing import X_SCALINGS, InputScalingMixin, TargetRangeClipMixin


class ClassicalGPRegressor(InputScalingMixin, TargetRangeClipMixin, BaseEstimator, RegressorMixin):
    """Classical Gaussian Process regressor with an RBF kernel.

    Uses an :math:`\\text{RBF}(l) + \\text{WhiteKernel}(\\sigma^2)` kernel
    inside a :class:`~sklearn.gaussian_process.GaussianProcessRegressor`.
    Targets are standardized before fitting so the kernel length scale has a
    consistent interpretation across datasets.

    ``x_scaling`` defaults to ``"standard"``, which is what this model hardcoded before the
    transform became a swept axis. It is swept now because the alternative is to tune the input
    transform for the matchgate kernels while leaving the baseline pinned, which would
    manufacture the advantage this benchmark exists to measure. The quantile transform in
    particular helps an RBF kernel for the same reason it helps the matchgate one: California
    Housing's heavy-tailed columns dominate a Euclidean distance just as they dominate a
    rotation angle.

    :param kernel_length_scale: Initial length scale :math:`l` for the RBF kernel.
    :param alpha: Value added to the diagonal of the Gram matrix for numerical
        stability (noise variance if not using a WhiteKernel).
    :param n_restarts_optimizer: Number of restarts for the kernel-hyperparameter
        optimizer. Set to 0 to skip optimization and use the initial values.
    :param x_scaling: Input transform, one of
        :data:`~fermiml_framework_regression.preprocessing.X_SCALINGS`.
    """

    MODEL_NAME = "ClassicalGP"
    DEFAULT_KERNEL_LENGTH_SCALE = 1.0
    DEFAULT_ALPHA = 1e-6
    DEFAULT_N_RESTARTS_OPTIMIZER = 0
    DEFAULT_X_SCALING = "standard"

    def __init__(
        self,
        *,
        kernel_length_scale: float = DEFAULT_KERNEL_LENGTH_SCALE,
        alpha: float = DEFAULT_ALPHA,
        n_restarts_optimizer: int = DEFAULT_N_RESTARTS_OPTIMIZER,
        x_scaling: str = DEFAULT_X_SCALING,
    ):
        self.kernel_length_scale = kernel_length_scale
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        if x_scaling is not None and x_scaling not in X_SCALINGS:
            raise ValueError(f"Unknown x_scaling {x_scaling!r}; expected one of {X_SCALINGS} or None.")
        self.x_scaling = x_scaling

    def fit(self, x_train: NDArray, y_train: NDArray) -> "ClassicalGPRegressor":
        """Fit the GP on training data.

        :param x_train: Training inputs of shape ``(n_samples, n_features)``.
        :param y_train: Training targets of shape ``(n_samples,)``.
        :return: The fitted regressor.
        """
        x_scaled = self._fit_x_scaler(x_train)
        self._fit_target_range(y_train)
        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y_train.reshape(-1, 1)).ravel()
        kernel = RBF(length_scale=self.kernel_length_scale) + WhiteKernel()
        self.gp_ = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts_optimizer,
        )
        self.gp_.fit(x_scaled, y_scaled)
        return self

    def predict(self, x: NDArray) -> NDArray:
        """Predict regression targets.

        :param x: Inputs of shape ``(n_samples, n_features)``.
        :return: Predicted targets of shape ``(n_samples,)``, clipped to the training target
            range.
        """
        y_scaled = self.gp_.predict(self._scale_x(x))
        return self._clip_to_target_range(self.y_scaler_.inverse_transform(y_scaled.reshape(-1, 1)).ravel())

    def predict_std(self, x: NDArray) -> NDArray:
        """Predictive standard deviation, in the original target units.

        Mirrors :meth:`NIFGPRegressor.predict_std` so a calibration figure can treat the
        classical and matchgate GPs identically. Both the input transform and the target scaler
        are applied.

        Not clipped, unlike :meth:`predict`: this is a spread, not a location, so bounding it by
        the training target range would be meaningless.

        :param x: Inputs of shape ``(n_samples, n_features)``.
        :return: Predictive standard deviations of shape ``(n_samples,)``.
        :rtype: NDArray
        """
        _, std_scaled = self.gp_.predict(self._scale_x(x), return_std=True)
        return np.asarray(std_scaled) * float(self.y_scaler_.scale_[0])

import warnings
from abc import abstractmethod

import numpy as np
from matchcake.ml.kernels import NIFKernel
from numpy._typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from ..nif_covariance_kernel import NIFCovarianceKernel
from ..preprocessing import X_SCALINGS, InputScalingMixin, TargetRangeClipMixin
from ..sklearn_nif_covariance_kernel import SklearnNIFCovarianceKernel


class NIFGPRegressor(InputScalingMixin, TargetRangeClipMixin, BaseEstimator, RegressorMixin):
    """Abstract base class for NIF Gaussian Process regressors.

    Handles the common GP fitting loop: target standardization, kernel
    construction via :meth:`_build_nif_kernel`, wrapping with
    :class:`SklearnNIFCovarianceKernel`, and fitting the
    :class:`~sklearn.gaussian_process.GaussianProcessRegressor`. Subclasses
    need only implement :meth:`_build_nif_kernel`.

    Practical notes:

    * Targets are standardized internally; do **not** pass pre-standardized
      targets. Inputs are passed through untouched unless ``x_scaling`` asks
      for a transform.
    * Setting ``normalize_y=True`` in the underlying GP ill-conditions the
      covariance kernel; this class always uses ``normalize_y=False``.
    * Predictions are clipped to the training target range; see
      :class:`~fermiml_framework_regression.preprocessing.TargetRangeClipMixin`.

    :param n_qubits: Number of qubits. ``-1`` infers from input dimension.
    :param x_scaling: Input transform applied before the embedding, one of
        :data:`~fermiml_framework_regression.preprocessing.X_SCALINGS`.
    :param sector_m: Even Majorana sector :math:`m` for the Pfaffian feature map.
    :param alpha: Noise level added to the diagonal of the Gram matrix. Raised by
        :attr:`JITTER_ESCALATION_FACTOR` and retried if the Cholesky factorization fails; the
        value actually used is exported as ``effective_alpha_``.
    :param scale: Extra multiplicative scale passed to
        :class:`SklearnNIFCovarianceKernel`.
    :param gram_batch_size: Batch size for Gram matrix assembly.
    :param random_state: Random seed for reproducibility.
    :param alignment: Whether to align the kernel on training data.
    :param alignment_iterations: Maximum alignment iterations.
    :param alignment_learning_rate: Learning rate for the alignment optimizer.
    :param alignment_early_stopping_patience: Patience for early stopping.
    :param alignment_early_stopping_threshold: Improvement threshold.
    """

    MODEL_NAME = "NIFGPRegressor"
    DEFAULT_N_QUBITS = -1
    DEFAULT_SECTOR_M = 2
    DEFAULT_ALPHA = 1e-4
    DEFAULT_SCALE = 1.0
    DEFAULT_X_SCALING = "none"
    JITTER_ESCALATION_FACTOR = 10.0
    JITTER_MAX_RETRIES = 4

    def __init__(
        self,
        *,
        n_qubits: int = DEFAULT_N_QUBITS,
        sector_m: int = DEFAULT_SECTOR_M,
        alpha: float = DEFAULT_ALPHA,
        scale: float = DEFAULT_SCALE,
        x_scaling: str = DEFAULT_X_SCALING,
        gram_batch_size: int = NIFKernel.DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = NIFKernel.DEFAULT_RANDOM_STATE,
        alignment: bool = NIFKernel.DEFAULT_ALIGNMENT,
        alignment_iterations: int = NIFKernel.DEFAULT_ALIGNMENT_ITERATIONS,
        alignment_learning_rate: float = NIFKernel.DEFAULT_ALIGNMENT_LEARNING_RATE,
        alignment_early_stopping_patience: int = NIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_PATIENCE,
        alignment_early_stopping_threshold: float = NIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_THRESHOLD,
    ):
        self.n_qubits = n_qubits
        self.sector_m = sector_m
        self.alpha = alpha
        self.scale = scale
        if x_scaling is not None and x_scaling not in X_SCALINGS:
            raise ValueError(f"Unknown x_scaling {x_scaling!r}; expected one of {X_SCALINGS} or None.")
        self.x_scaling = x_scaling
        self.gram_batch_size = gram_batch_size
        self.random_state = random_state
        self.alignment = alignment
        self.alignment_iterations = alignment_iterations
        self.alignment_learning_rate = alignment_learning_rate
        self.alignment_early_stopping_patience = alignment_early_stopping_patience
        self.alignment_early_stopping_threshold = alignment_early_stopping_threshold

    @abstractmethod
    def _build_nif_kernel(self, n_qubits: int) -> NIFCovarianceKernel:
        """Construct and return the concrete NIF covariance kernel.

        :param n_qubits: Resolved number of qubits (never ``-1``).
        :return: An unfitted :class:`NIFCovarianceKernel` subclass instance.
        """

    def fit(self, x_train: NDArray, y_train: NDArray) -> "NIFGPRegressor":
        """Fit the NIF kernel and the Gaussian Process on training data.

        :param x_train: Training inputs of shape ``(n_samples, n_features)``.
        :param y_train: Training targets of shape ``(n_samples,)``.
        :return: The fitted regressor.
        """
        n_qubits = x_train.shape[1] if self.n_qubits < 0 else self.n_qubits
        # Both fitted on the training split only, so neither carries test information.
        x_scaled = self._fit_x_scaler(x_train)
        self._fit_target_range(y_train)
        self.y_scaler_ = StandardScaler()
        y_scaled = self.y_scaler_.fit_transform(y_train.reshape(-1, 1)).ravel()
        self.nif_kernel_ = self._build_nif_kernel(n_qubits)
        self.nif_kernel_.fit(x_scaled, y_scaled)
        sklearn_kernel = SklearnNIFCovarianceKernel(self.nif_kernel_, scale=self.scale)
        self._fit_gp_with_jitter(sklearn_kernel, x_scaled, y_scaled)
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

        The posterior variance is the one thing a Gaussian process offers that a support-vector
        regressor does not, so it is worth recording alongside the point predictions: it is what
        a calibration figure needs, and it cannot be reconstructed after the fact.

        The underlying GP is fitted on standardized targets, so its standard deviation is
        rescaled by the target scaler before being returned.

        :param x: Inputs of shape ``(n_samples, n_features)``.
        :return: Predictive standard deviations of shape ``(n_samples,)``.
        :rtype: NDArray
        """
        _, std_scaled = self.gp_.predict(self._scale_x(x), return_std=True)
        return np.asarray(std_scaled) * float(self.y_scaler_.scale_[0])

    def _fit_gp_with_jitter(self, sklearn_kernel: SklearnNIFCovarianceKernel, x_train: NDArray, y_scaled: NDArray):
        """Fit the GP, raising the diagonal jitter until the Cholesky factorization succeeds.

        The requested ``alpha`` is always tried first and is what almost every configuration
        runs at, so this changes no result that was already obtainable: the retry path is
        entered only for a fit that would otherwise raise. What it buys is that a Gram matrix
        sitting a rounding error away from indefinite costs a slightly larger jitter instead of
        an entire lost job.

        The escalation is recorded rather than silent. A cell fitted at a larger alpha than its
        configuration names is not comparable to its neighbours without saying so, so
        ``effective_alpha_`` and ``n_jitter_retries_`` are exported for the run output and a
        warning is emitted into the job log.

        :param sklearn_kernel: The kernel wrapper handed to the GP.
        :param x_train: Training inputs of shape ``(n_samples, n_features)``.
        :param y_scaled: Standardized training targets of shape ``(n_samples,)``.
        :return: The fitted regressor.
        :rtype: NIFGPRegressor
        """
        alpha = float(self.alpha)
        for retry in range(self.JITTER_MAX_RETRIES + 1):
            self.gp_ = GaussianProcessRegressor(
                kernel=sklearn_kernel,
                alpha=alpha,
                normalize_y=False,
                optimizer=None,
            )
            try:
                self.gp_.fit(x_train, y_scaled)
            except np.linalg.LinAlgError:
                if retry == self.JITTER_MAX_RETRIES:
                    raise
                alpha *= self.JITTER_ESCALATION_FACTOR
                continue
            self.effective_alpha_ = alpha
            self.n_jitter_retries_ = retry
            if retry:
                warnings.warn(
                    f"{type(self).__name__}: the Gram matrix was not positive definite at "
                    f"alpha={self.alpha:g}; fitted at alpha={alpha:g} after {retry} jitter "
                    f"escalation(s). This cell is regularized more strongly than its configuration "
                    f"asks for; see effective_alpha in the run output.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return self
        raise AssertionError("unreachable: the loop returns or re-raises on its last iteration")

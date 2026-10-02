from typing import Optional, Union

import torch
from matchcake.ml.kernels.nif_kernel import NIFKernel
from matchcake.typing import TensorLike
from matchcake.utils.torch_utils import to_numpy, to_tensor
from numpy._typing import NDArray
from sklearn.base import RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ..preprocessing import X_SCALINGS, InputScalingMixin, TargetRangeClipMixin


class NIFKernelRegressor(InputScalingMixin, TargetRangeClipMixin, NIFKernel, RegressorMixin):
    """Base NIF support-vector regressor.

    Wraps a :class:`~matchcake.ml.kernels.nif_kernel.NIFKernel` subclass as a
    sklearn-compatible regressor by pairing it with an :class:`~sklearn.svm.SVR`
    using a precomputed kernel. Inputs pass through the transform named by
    ``x_scaling`` and targets are standardized before fitting.

    Subclasses must implement :meth:`ansatz` (inherited from
    :class:`NIFKernel`).

    :param gram_batch_size: Batch size for Gram matrix assembly.
    :param random_state: Random seed for reproducibility.
    :param alignment: Whether to perform kernel alignment during :meth:`fit`.
    :param alignment_iterations: Maximum alignment iterations.
    :param alignment_learning_rate: Learning rate for the alignment optimizer.
    :param alignment_early_stopping_patience: Patience for alignment early stopping.
    :param alignment_early_stopping_threshold: Improvement threshold for alignment.
    :param svr_c: Regularisation parameter for :class:`~sklearn.svm.SVR`.
    :param svr_epsilon: Epsilon tube half-width in standardized target units.
    :param n_qubits: Number of qubits. Use ``-1`` to infer from input dimension.
    :param x_scaling: Input transform, one of
        :data:`~fermiml_framework_regression.preprocessing.X_SCALINGS`.
    """

    MODEL_NAME = "NIFSVRKernel"
    DEFAULT_SVR_C = 1.0
    DEFAULT_SVR_EPSILON = 0.1
    DEFAULT_X_SCALING = "minmax11"

    def __init__(
        self,
        *,
        gram_batch_size: int = NIFKernel.DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = NIFKernel.DEFAULT_RANDOM_STATE,
        alignment: bool = NIFKernel.DEFAULT_ALIGNMENT,
        alignment_iterations: int = NIFKernel.DEFAULT_ALIGNMENT_ITERATIONS,
        alignment_learning_rate: float = NIFKernel.DEFAULT_ALIGNMENT_LEARNING_RATE,
        alignment_early_stopping_patience: int = NIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_PATIENCE,
        alignment_early_stopping_threshold: float = NIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_THRESHOLD,
        svr_c: float = DEFAULT_SVR_C,
        svr_epsilon: float = DEFAULT_SVR_EPSILON,
        n_qubits: int = NIFKernel.DEFAULT_N_QUBITS,
        x_scaling: str = DEFAULT_X_SCALING,
    ):
        super().__init__(
            gram_batch_size=gram_batch_size,
            random_state=random_state,
            alignment=alignment,
            alignment_iterations=alignment_iterations,
            alignment_learning_rate=alignment_learning_rate,
            alignment_early_stopping_patience=alignment_early_stopping_patience,
            alignment_early_stopping_threshold=alignment_early_stopping_threshold,
            n_qubits=n_qubits,
        )
        if x_scaling is not None and x_scaling not in X_SCALINGS:
            raise ValueError(f"Unknown x_scaling {x_scaling!r}; expected one of {X_SCALINGS} or None.")
        self.x_scaling = x_scaling
        self.y_scaler = StandardScaler()
        self.svr_c = svr_c
        self.svr_epsilon = svr_epsilon
        self.svr = SVR(kernel="precomputed")

    def fit(self, x_train: TensorLike, y_train: Optional[TensorLike] = None) -> "NIFKernelRegressor":
        """Fit the kernel and the SVR on training data.

        The parent is fitted on the **scaled** inputs, which is what makes the kernel
        self-consistent. ``Kernel.fit`` stores its argument as ``x_train_``, and ``transform(x)``
        evaluates ``K(x, x_train_)``; since ``predict`` scales its query, handing the parent the
        raw inputs here would fit the SVR on ``K(scaled, raw)`` and leave the reference set in
        different units from every query. It is silent -- the numbers stay finite and only the
        accuracy suffers -- and it was measured at 9.03 in prediction units on a 4-feature
        fixture, largest for the transforms that move the inputs furthest.

        :param x_train: Training inputs of shape ``(n_samples, n_features)``.
        :param y_train: Training targets of shape ``(n_samples,)``.
        :return: The fitted regressor.
        """
        x_train = to_numpy(x_train)
        y_train = to_numpy(y_train)
        x_scaled = self._fit_x_scaler(x_train)
        self._fit_target_range(y_train)
        self.y_scaler.fit(y_train.reshape(-1, 1))
        super().fit(x_scaled, y_train)
        self.svr.C = self.svr_c
        self.svr.epsilon = self.svr_epsilon
        y_scaled = self.y_scaler.transform(y_train.reshape(-1, 1)).ravel()
        x_kernel = to_numpy(self.transform(to_tensor(x_scaled, dtype=self.R_DTYPE, device=self.device)))
        self.svr.fit(x_kernel, y_scaled)
        return self

    def predict(self, x: Union[NDArray, TensorLike]) -> NDArray:
        """Predict regression targets for inputs ``x``.

        :param x: Inputs of shape ``(n_samples, n_features)``.
        :return: Predicted targets of shape ``(n_samples,)``, clipped to the training target
            range.
        """
        self.eval()
        with torch.no_grad():
            x_scaled = self._scale_x(to_numpy(x))
            x_tensor = to_tensor(x_scaled, dtype=self.R_DTYPE, device=self.device)
            k_test = to_numpy(self.transform(x_tensor))
            y_scaled = self.svr.predict(k_test)
        return self._clip_to_target_range(self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel())

from typing import Optional

import torch
from matchcake.ml import LinearNIFKernel
from matchcake.typing import TensorLike
from matchcake.utils.torch_utils import to_numpy
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ..preprocessing import X_SCALINGS
from .nif_kernel_regressor import NIFKernelRegressor


class VanillaNIFKernelRegressor(NIFKernelRegressor, LinearNIFKernel):
    """NIF support-vector regressor with the linear (Vanilla) ansatz.

    Combines :class:`NIFKernelRegressor` with :class:`LinearNIFKernel`:
    the input is pushed through a single :class:`torch.nn.LazyLinear` layer
    whose output fills the upper triangle of an antisymmetric
    :math:`2n \\times 2n` Hamiltonian generator, which is exponentiated into
    one global SPTM. The resulting fidelity kernel is used with an SVR.

    Set ``n_qubits=-1`` to auto-detect the number of qubits from the input
    dimension at fit time.

    The two parents take disjoint ``__init__`` kwargs (``bias``/
    ``encoder_activation`` vs ``svr_c``/``svr_epsilon``), so this class
    calls :meth:`LinearNIFKernel.__init__` directly and then sets the SVR
    attributes by hand -- the same pattern used by
    :class:`~fermiml_framework_regression.VanillaNIFCovarianceKernel`.

    :param n_qubits: Number of qubits. ``-1`` infers from input dimension.
    :param svr_c: Regularisation parameter for :class:`~sklearn.svm.SVR`.
    :param svr_epsilon: Epsilon tube half-width in standardized target units.
    :param bias: Whether to include a bias in the :class:`~torch.nn.LazyLinear`.
    :param encoder_activation: Activation applied after the linear layer.
    :param x_scaling: Input transform, one of
        :data:`~fermiml_framework_regression.preprocessing.X_SCALINGS`.
    """

    MODEL_NAME = "VanillaSVR"
    PLACEHOLDER_N_QUBITS = 2

    def __init__(
        self,
        *,
        n_qubits: int = LinearNIFKernel.DEFAULT_N_QUBITS,
        svr_c: float = NIFKernelRegressor.DEFAULT_SVR_C,
        svr_epsilon: float = NIFKernelRegressor.DEFAULT_SVR_EPSILON,
        bias: bool = LinearNIFKernel.DEFAULT_BIAS,
        encoder_activation: str = LinearNIFKernel.DEFAULT_ENCODER_ACTIVATION,
        gram_batch_size: int = LinearNIFKernel.DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = LinearNIFKernel.DEFAULT_RANDOM_STATE,
        alignment: bool = LinearNIFKernel.DEFAULT_ALIGNMENT,
        alignment_iterations: int = LinearNIFKernel.DEFAULT_ALIGNMENT_ITERATIONS,
        alignment_learning_rate: float = LinearNIFKernel.DEFAULT_ALIGNMENT_LEARNING_RATE,
        alignment_early_stopping_patience: int = LinearNIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_PATIENCE,
        alignment_early_stopping_threshold: float = LinearNIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_THRESHOLD,
        x_scaling: str = NIFKernelRegressor.DEFAULT_X_SCALING,
    ):
        self._auto_n_qubits = n_qubits < 0
        LinearNIFKernel.__init__(
            self,
            n_qubits=self.PLACEHOLDER_N_QUBITS if self._auto_n_qubits else n_qubits,
            bias=bias,
            encoder_activation=encoder_activation,
            gram_batch_size=gram_batch_size,
            random_state=random_state,
            alignment=alignment,
            alignment_iterations=alignment_iterations,
            alignment_learning_rate=alignment_learning_rate,
            alignment_early_stopping_patience=alignment_early_stopping_patience,
            alignment_early_stopping_threshold=alignment_early_stopping_threshold,
        )
        if x_scaling is not None and x_scaling not in X_SCALINGS:
            raise ValueError(f"Unknown x_scaling {x_scaling!r}; expected one of {X_SCALINGS} or None.")
        self.x_scaling = x_scaling
        self.y_scaler = StandardScaler()
        self.svr_c = svr_c
        self.svr_epsilon = svr_epsilon
        self.svr = SVR(kernel="precomputed")

    def fit(self, x_train: TensorLike, y_train: Optional[TensorLike] = None) -> "VanillaNIFKernelRegressor":
        """Fit the regressor, auto-detecting qubits from input dimension when ``n_qubits < 0``.

        ``y_train`` keeps the optional signature of :meth:`NIFKernelRegressor.fit` rather than
        narrowing it to required: a subclass that demands more than its parent cannot stand in
        for it, and mypy rejects the override. The parent still needs a real target, so passing
        None fails there, as it always has.

        :param x_train: Training inputs of shape ``(n_samples, n_features)``.
        :param y_train: Training targets of shape ``(n_samples,)``.
        :return: The fitted regressor.
        :rtype: VanillaNIFKernelRegressor
        """
        torch.manual_seed(self.random_state)
        x_arr = to_numpy(x_train)
        if self._auto_n_qubits:
            self.n_qubits = x_arr.shape[-1]
        return NIFKernelRegressor.fit(self, x_arr, y_train)

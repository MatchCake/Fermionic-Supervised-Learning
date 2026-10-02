import torch
from matchcake.ml import LinearNIFKernel
from numpy._typing import NDArray

from ..nif_covariance_kernel import NIFCovarianceKernel
from ..vanilla_nif_covariance_kernel import VanillaNIFCovarianceKernel
from .nif_gp_regressor import NIFGPRegressor


class VanillaNIFGPRegressor(NIFGPRegressor):
    """NIF Gaussian Process regressor with the trainable linear (Vanilla) ansatz.

    Uses :class:`VanillaNIFCovarianceKernel` as the covariance kernel. The
    linear ansatz uses a :class:`torch.nn.LazyLinear` encoder that materializes
    its weights on the first forward pass during :meth:`fit`.

    Set ``n_qubits=-1`` to auto-detect from the input dimension.

    :param n_qubits: Number of qubits. ``-1`` infers from input dimension.
    :param sector_m: Even Majorana sector for the Pfaffian feature map.
    :param bias: Whether the :class:`~torch.nn.LazyLinear` encoder has a bias.
    :param encoder_activation: Activation applied after the linear layer.
    :param alpha: GP noise diagonal.
    :param scale: Extra scale for :class:`SklearnNIFCovarianceKernel`.
    :param x_scaling: Input transform applied before the embedding, one of
        :data:`~fermiml_framework_regression.preprocessing.X_SCALINGS`.
    """

    MODEL_NAME = "VanillaNIFGP"
    DEFAULT_BIAS = LinearNIFKernel.DEFAULT_BIAS
    DEFAULT_ENCODER_ACTIVATION = LinearNIFKernel.DEFAULT_ENCODER_ACTIVATION

    def __init__(
        self,
        *,
        n_qubits: int = NIFGPRegressor.DEFAULT_N_QUBITS,
        sector_m: int = NIFGPRegressor.DEFAULT_SECTOR_M,
        bias: bool = DEFAULT_BIAS,
        encoder_activation: str = DEFAULT_ENCODER_ACTIVATION,
        alpha: float = NIFGPRegressor.DEFAULT_ALPHA,
        scale: float = NIFGPRegressor.DEFAULT_SCALE,
        x_scaling: str = NIFGPRegressor.DEFAULT_X_SCALING,
        gram_batch_size: int = LinearNIFKernel.DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = LinearNIFKernel.DEFAULT_RANDOM_STATE,
        alignment: bool = LinearNIFKernel.DEFAULT_ALIGNMENT,
        alignment_iterations: int = LinearNIFKernel.DEFAULT_ALIGNMENT_ITERATIONS,
        alignment_learning_rate: float = LinearNIFKernel.DEFAULT_ALIGNMENT_LEARNING_RATE,
        alignment_early_stopping_patience: int = LinearNIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_PATIENCE,
        alignment_early_stopping_threshold: float = LinearNIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_THRESHOLD,
    ):
        super().__init__(
            n_qubits=n_qubits,
            sector_m=sector_m,
            alpha=alpha,
            scale=scale,
            x_scaling=x_scaling,
            gram_batch_size=gram_batch_size,
            random_state=random_state,
            alignment=alignment,
            alignment_iterations=alignment_iterations,
            alignment_learning_rate=alignment_learning_rate,
            alignment_early_stopping_patience=alignment_early_stopping_patience,
            alignment_early_stopping_threshold=alignment_early_stopping_threshold,
        )
        self.bias = bias
        self.encoder_activation = encoder_activation

    def _build_nif_kernel(self, n_qubits: int) -> NIFCovarianceKernel:
        """Build a :class:`VanillaNIFCovarianceKernel` for the given qubit count.

        The ansatz is a :class:`torch.nn.LazyLinear` whose weights materialise from torch's
        global RNG on the first forward pass. ``random_state`` does not reach that RNG, so
        without seeding it here two runs of the same configuration disagree -- which for a
        hash-addressed benchmark means a requeued job can overwrite a result with a different
        number. Seeded at construction, before anything can trigger materialisation.

        :param n_qubits: Resolved number of qubits.
        :return: Unfitted :class:`VanillaNIFCovarianceKernel` instance.
        """
        torch.manual_seed(self.random_state)
        return VanillaNIFCovarianceKernel(
            n_qubits=n_qubits,
            sector_m=self.sector_m,
            bias=self.bias,
            encoder_activation=self.encoder_activation,
            gram_batch_size=self.gram_batch_size,
            random_state=self.random_state,
            alignment=self.alignment,
            alignment_iterations=self.alignment_iterations,
            alignment_learning_rate=self.alignment_learning_rate,
            alignment_early_stopping_patience=self.alignment_early_stopping_patience,
            alignment_early_stopping_threshold=self.alignment_early_stopping_threshold,
        )

    def fit(self, x_train: NDArray, y_train: NDArray) -> "VanillaNIFGPRegressor":
        """Fit the regressor on training data.

        :param x_train: Training inputs of shape ``(n_samples, n_features)``.
        :param y_train: Training targets of shape ``(n_samples,)``.
        :return: The fitted regressor.
        """
        return super().fit(x_train, y_train)  # type: ignore[return-value]

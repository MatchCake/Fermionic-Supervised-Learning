from matchcake.ml.kernels import NIFKernel
from numpy._typing import NDArray

from ..angle_nif_covariance_kernel import AngleNIFCovarianceKernel
from ..nif_covariance_kernel import NIFCovarianceKernel
from .nif_gp_regressor import NIFGPRegressor


class AngleNIFGPRegressor(NIFGPRegressor):
    """NIF Gaussian Process regressor with a fixed angle-embedding ansatz.

    Uses :class:`AngleNIFCovarianceKernel` (no trainable parameters) as the
    covariance kernel. The angle-embedding ansatz expects ``n_qubits`` to be
    even when the number of input features equals ``n_qubits``; auto-detection
    (``n_qubits=-1``) sets it to the input dimension, so datasets with an odd
    feature count require an explicit even ``n_qubits``.

    :param n_qubits: Number of qubits. ``-1`` infers from input dimension.
    :param sector_m: Even Majorana sector for the Pfaffian feature map.
    :param rotations: Rotation layers for the angle embedding, e.g. ``"Y,X"``.
    :param alpha: GP noise diagonal.
    :param scale: Extra scale for :class:`SklearnNIFCovarianceKernel`.
    :param x_scaling: Input transform applied before the embedding, one of
        :data:`~fermiml_framework_regression.preprocessing.X_SCALINGS`.
    """

    MODEL_NAME = "AngleNIFGP"
    DEFAULT_ROTATIONS = AngleNIFCovarianceKernel.DEFAULT_ROTATIONS

    def __init__(
        self,
        *,
        n_qubits: int = NIFGPRegressor.DEFAULT_N_QUBITS,
        sector_m: int = NIFGPRegressor.DEFAULT_SECTOR_M,
        rotations: str = DEFAULT_ROTATIONS,
        alpha: float = NIFGPRegressor.DEFAULT_ALPHA,
        scale: float = NIFGPRegressor.DEFAULT_SCALE,
        x_scaling: str = NIFGPRegressor.DEFAULT_X_SCALING,
        gram_batch_size: int = NIFKernel.DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = NIFKernel.DEFAULT_RANDOM_STATE,
        alignment: bool = NIFKernel.DEFAULT_ALIGNMENT,
        alignment_iterations: int = NIFKernel.DEFAULT_ALIGNMENT_ITERATIONS,
        alignment_learning_rate: float = NIFKernel.DEFAULT_ALIGNMENT_LEARNING_RATE,
        alignment_early_stopping_patience: int = NIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_PATIENCE,
        alignment_early_stopping_threshold: float = NIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_THRESHOLD,
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
        self.rotations = rotations

    def _build_nif_kernel(self, n_qubits: int) -> NIFCovarianceKernel:
        """Build an :class:`AngleNIFCovarianceKernel` for the given qubit count.

        :param n_qubits: Resolved number of qubits.
        :return: Unfitted :class:`AngleNIFCovarianceKernel` instance.
        """
        return AngleNIFCovarianceKernel(
            n_qubits=n_qubits,
            sector_m=self.sector_m,
            rotations=self.rotations,
            gram_batch_size=self.gram_batch_size,
            random_state=self.random_state,
            alignment=self.alignment,
            alignment_iterations=self.alignment_iterations,
            alignment_learning_rate=self.alignment_learning_rate,
            alignment_early_stopping_patience=self.alignment_early_stopping_patience,
            alignment_early_stopping_threshold=self.alignment_early_stopping_threshold,
        )

    def fit(self, x_train: NDArray, y_train: NDArray) -> "AngleNIFGPRegressor":
        """Fit the regressor on training data.

        :param x_train: Training inputs of shape ``(n_samples, n_features)``.
        :param y_train: Training targets of shape ``(n_samples,)``.
        :return: The fitted regressor.
        """
        return super().fit(x_train, y_train)  # type: ignore[return-value]

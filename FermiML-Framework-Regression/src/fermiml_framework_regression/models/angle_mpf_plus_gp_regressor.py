from typing import Sequence

from matchcake.ml.kernels import NIFKernel
from numpy._typing import NDArray

from ..angle_mpf_plus_covariance_kernel import AngleMpfPlusCovarianceKernel
from ..angle_nif_covariance_kernel import AngleNIFCovarianceKernel
from ..multi_sector_plus_kernel import MultiSectorPlusCovarianceKernel
from ..nif_covariance_kernel import NIFCovarianceKernel
from .nif_gp_regressor import NIFGPRegressor


class AngleMpfPlusGPRegressor(NIFGPRegressor):
    r"""Multi-sector covariance-kernel GP with the fixed angle-embedding ansatz.

    :class:`AngleNIFGPRegressor` with the feature map of
    :class:`~fermiml_framework_regression.multi_sector_plus_kernel.MultiSectorPlusCovarianceKernel`
    substituted for the single-order one: the circuit starts from :math:`\ket{+}^{\otimes n}`
    and the features span every order in ``sectors`` rather than the one order ``sector_m``.
    The ansatz keeps no trainable parameters.

    ``sector_m`` is therefore not a parameter here; ``sectors`` replaces it.

    :param n_qubits: Number of qubits. ``-1`` infers from input dimension.
    :param sectors: Majorana orders to span, e.g. ``(1, 2, 3)``.
    :param rotations: Comma-separated rotation letters for the angle embedding.
    :param alpha: GP noise diagonal.
    :param scale: Extra scale for :class:`SklearnNIFCovarianceKernel`.
    :param x_scaling: Input transform applied before the embedding, one of
        :data:`~fermiml_framework_regression.preprocessing.X_SCALINGS`.
    """

    MODEL_NAME = "AngleMpfPlus"
    DEFAULT_SECTORS = MultiSectorPlusCovarianceKernel.DEFAULT_SECTORS
    DEFAULT_ROTATIONS = AngleNIFCovarianceKernel.DEFAULT_ROTATIONS

    def __init__(
        self,
        *,
        n_qubits: int = NIFGPRegressor.DEFAULT_N_QUBITS,
        sectors: Sequence[int] = DEFAULT_SECTORS,
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
        self.sectors = sectors
        self.rotations = rotations

    def _build_nif_kernel(self, n_qubits: int) -> NIFCovarianceKernel:
        """Build an :class:`AngleMpfPlusCovarianceKernel` for the given qubit count.

        :param n_qubits: Resolved number of qubits.
        :return: Unfitted :class:`AngleMpfPlusCovarianceKernel` instance.
        """
        return AngleMpfPlusCovarianceKernel(
            n_qubits=n_qubits,
            sectors=self.sectors,
            rotations=self.rotations,
            gram_batch_size=self.gram_batch_size,
            random_state=self.random_state,
            alignment=self.alignment,
            alignment_iterations=self.alignment_iterations,
            alignment_learning_rate=self.alignment_learning_rate,
            alignment_early_stopping_patience=self.alignment_early_stopping_patience,
            alignment_early_stopping_threshold=self.alignment_early_stopping_threshold,
        )

    def fit(self, x_train: NDArray, y_train: NDArray) -> "AngleMpfPlusGPRegressor":
        """Fit the regressor on training data.

        :param x_train: Training inputs of shape ``(n_samples, n_features)``.
        :param y_train: Training targets of shape ``(n_samples,)``.
        :return: The fitted regressor.
        """
        return super().fit(x_train, y_train)  # type: ignore[return-value]

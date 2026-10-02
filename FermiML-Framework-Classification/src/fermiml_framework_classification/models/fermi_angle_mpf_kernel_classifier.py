from functools import cached_property
from itertools import combinations
from typing import Optional

import numpy as np
import pennylane as qml
import torch
from matchcake.ml.kernels import NIFKernel
from matchcake.operations import SptmAngleEmbedding
from matchcake.typing import TensorLike
from matchcake.utils import sector_pfaffian_features
from matchcake.utils.torch_utils import to_numpy, to_tensor

from .nif_kernel_classifier import NIFKernelClassifier


class FermiAngleMpfKernelClassifier(NIFKernelClassifier):
    """SVM classifier using the matchgate Pfaffian covariance kernel with a fixed angle ansatz.

    Analogue of :class:`FermiVanillaMpfKernelClassifier` where the ansatz is the
    fixed :class:`~matchcake.operations.SptmAngleEmbedding` rather than a
    trainable linear layer. Input features are embedded directly as rotation
    angles, so there are no learnable parameters and no alignment step.

    The kernel value is the rescaled inner product of Pfaffian feature vectors:

    .. math::

        \\kappa_m(x, x') = \\frac{c}{2^n}\\, v_m(x)^\\top v_m(x'),
        \\qquad
        v_m(x)_S = \\mathrm{Pf}\\!\\bigl(\\Lambda(x)|_S\\bigr).

    Because the ansatz pairs features to adjacent wires, ``n_qubits`` must be
    even when it equals the number of input features.

    :param gram_batch_size: Batch size for Gram matrix computation.
    :param random_state: Seed for reproducibility.
    :param svm_c: Regularisation parameter of the SVC.
    :param sector_m: Even Majorana sector order :math:`m`. Must satisfy
        ``2 <= sector_m <= 2 * n_qubits - 2``. Defaults to 2.
    :param observable_norm: Multiplicative scale :math:`c`. Defaults to 1.0.
    :param n_qubits: Number of qubits. Pass -1 (default) to infer from the
        feature dimension at fit time.
    :param rotations: Comma-separated rotation letters for the angle embedding,
        e.g. ``\"Y,X\"``. Passed to
        :class:`~matchcake.operations.SptmAngleEmbedding`.
    """

    MODEL_NAME = "AngleMpf"
    DEFAULT_SECTOR_M = 2
    DEFAULT_OBSERVABLE_NORM = 1.0
    DEFAULT_ROTATIONS = "Y,X"
    MAX_PFAFFIAN_FEATURE_TERMS = 10_000_000  # C(2n, m) above this OOMs (~few GB)

    def __init__(
        self,
        *,
        gram_batch_size: int = NIFKernelClassifier.DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = NIFKernelClassifier.DEFAULT_RANDOM_STATE,
        svm_c: float = NIFKernelClassifier.DEFAULT_SVM_C,
        sector_m: int = DEFAULT_SECTOR_M,
        observable_norm: float = DEFAULT_OBSERVABLE_NORM,
        n_qubits: int = -1,
        rotations: str = DEFAULT_ROTATIONS,
    ):
        self._n_qubits_is_default_flag = False
        if n_qubits is None or n_qubits < 0:
            self._n_qubits_is_default_flag = True
            n_qubits = 2
        super().__init__(
            gram_batch_size=gram_batch_size,
            random_state=random_state,
            svm_c=svm_c,
            n_qubits=n_qubits,
        )
        if sector_m % 2 != 0 or sector_m < 2:
            raise ValueError(f"sector_m must be a positive even integer; got {sector_m}.")
        if not self._n_qubits_is_default_flag and sector_m > 2 * self.n_qubits - 2:
            raise ValueError(
                f"sector_m={sector_m} exceeds 2n-2={2 * self.n_qubits - 2}; " f"no non-trivial m-subsets exist."
            )
        self._sector_m = sector_m
        self.observable_norm = observable_norm
        self.rotations = rotations

    def fit(self, x_train: TensorLike, y_train: Optional[TensorLike] = None):
        """Fit the classifier on training data.

        :param x_train: Training features of shape ``(n_samples, n_features)``.
        :param y_train: Training labels of shape ``(n_samples,)``.
        :return: The fitted classifier.
        :rtype: FermiAngleMpfKernelClassifier
        """
        if self._n_qubits_is_default_flag:
            n_features = qml.math.shape(x_train)[-1]
            # matchgate circuits require an even number of wires
            self.n_qubits = n_features if n_features % 2 == 0 else n_features + 1
        if self._sector_m > 2 * self.n_qubits - 2:
            raise ValueError(
                f"sector_m={self._sector_m} exceeds 2n-2={2 * self.n_qubits - 2}; " f"no non-trivial m-subsets exist."
            )
        from math import comb

        n_terms = comb(2 * self.n_qubits, self._sector_m)
        if n_terms > self.MAX_PFAFFIAN_FEATURE_TERMS:
            raise MemoryError(
                f"Pfaffian feature dimension C(2*{self.n_qubits},{self._sector_m})={n_terms:,} "
                f"exceeds limit {self.MAX_PFAFFIAN_FEATURE_TERMS:,}. Reduce n_qubits or sector_m."
            )
        return super().fit(x_train, y_train)

    def ansatz(self, x: TensorLike):
        """Angle-embedding ansatz: input features become rotation angles.

        :param x: Input batch of shape ``(batch_size, n_features)``.
        :return: Generator of matchgate operations.
        """
        yield from SptmAngleEmbedding(x, wires=self.wires, rotations=self.rotations).decomposition()

    def compute_similarities(self, x0: torch.Tensor, x1: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the Pfaffian covariance Gram matrix between x0 and x1.

        :param x0: First input tensor of shape ``(n_samples_0, n_features)``.
        :param x1: Second input tensor of shape ``(n_samples_1, n_features)``.
        :return: Gram matrix of shape ``(n_samples_0, n_samples_1)``.
        :rtype: torch.Tensor
        """
        v0 = self.feature_map(x0)
        if kwargs.get("x1_train", False):
            v1 = self.features_train
        elif x1 is x0:
            v1 = v0
        else:
            v1 = self.feature_map(x1)
        prefactor = self.observable_norm / (2**self.n_qubits)
        gram = prefactor * torch.einsum("if,jf->ij", v0, v1)
        return gram.to(dtype=self.R_DTYPE)

    def feature_map(self, x: TensorLike) -> torch.Tensor:
        """Compute the order-m Pfaffian feature map.

        :param x: Input tensor of shape ``(n_samples, n_features)``.
        :return: Tensor of shape ``(n_samples, n_terms)`` containing the
            signed Pfaffian features for each sample.
        :rtype: torch.Tensor
        """
        x = to_tensor(x, dtype=self.R_DTYPE, device=self.device)  # type: ignore
        features = []
        for start in range(0, len(x), self.gram_batch_size):
            batch = x[start : start + self.gram_batch_size]
            self._q_device.execute_generator(self.ansatz(batch), reset=True)
            lambda_x = self._q_device.covariance_matrix  # (B, 2n, 2n)
            v = sector_pfaffian_features(lambda_x, self.index_sets)  # (B, n_terms)
            features.append(to_tensor(v, dtype=self.R_DTYPE, device=self.device))  # type: ignore
        return torch.cat(features, dim=0)

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        return len(self._q_device.wires)

    @n_qubits.setter
    def n_qubits(self, value: int) -> None:
        # Rebuild the quantum device, then invalidate caches that depend on n_qubits.
        NIFKernel.n_qubits.fset(self, value)  # type: ignore[union-attr]
        self.__dict__.pop("index_sets", None)
        self.__dict__.pop("features_train", None)

    @property
    def sector_m(self) -> int:
        """Even Majorana sector order m."""
        return self._sector_m

    @sector_m.setter
    def sector_m(self, value: int) -> None:
        if value % 2 != 0 or value < 2:
            raise ValueError(f"sector_m must be a positive even integer; got {value}.")
        self._sector_m = value
        self.__dict__.pop("index_sets", None)

    @property
    def n_terms(self) -> int:
        """Dimension of the Pfaffian feature map, C(2n, m)."""
        return int(self.index_sets.shape[0])

    @cached_property
    def index_sets(self) -> np.ndarray:
        """All C(2n, m) ordered m-subsets of the 2n Majorana indices."""
        return np.array(
            list(combinations(range(2 * self.n_qubits), self._sector_m)),
            dtype=np.int64,
        )

    @cached_property
    def features_train(self) -> torch.Tensor:
        """Cached Pfaffian feature map of the scaled training set."""
        x = to_tensor(self.scaler.transform(to_numpy(self.x_train_)), dtype=self.R_DTYPE)
        return self.feature_map(x)

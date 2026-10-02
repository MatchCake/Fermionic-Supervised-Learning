from functools import cached_property
from itertools import combinations

import numpy as np
import torch
from matchcake.ml.kernels import NIFKernel
from matchcake.typing import TensorLike
from matchcake.utils import sector_pfaffian_features
from matchcake.utils.torch_utils import to_tensor


class NIFCovarianceKernel(NIFKernel):
    r"""Base matchgate covariance kernel (arXiv:2605.00099).

    For two matchgate-evolved fermionic Gaussian states :math:`\rho(x)` and
    :math:`\rho(x')` with Majorana covariance matrices :math:`\Lambda(x)` and
    :math:`\Lambda(x')`, the kernel is the rescaled Euclidean inner product of
    the order-:math:`m` Pfaffian feature maps:

    .. math::

        \kappa_m(x, x') = \frac{c}{2^n}\, v_m(x)^\top v_m(x'),
        \qquad
        v_m(x)_S = \mathrm{Pf}\!\bigl(\Lambda(x)|_S\bigr),

    where :math:`S` ranges over all :math:`\binom{2n}{m}` ordered
    :math:`m`-subsets of the :math:`2n` Majorana indices and :math:`c` is the
    ``observable_norm`` scale (the :math:`\lVert O \rVert^2` prefactor of the
    paper, defaulting to 1).

    Subclasses must implement :meth:`ansatz` to define the matchgate circuit
    :math:`U(x)` that prepares :math:`\rho(x)`. Everything else -- the
    order-:math:`m` Pfaffian feature map and the Gram assembly -- is provided
    here.

    Two differences from the fidelity kernels in this package:

    * The diagonal :math:`\kappa_m(x, x) = \tfrac{c}{2^n}\lVert v_m(x)\rVert^2`
      is **not** identically 1. The base-class
      :class:`~matchcake.ml.kernels.gram_matrix.GramMatrix` hard-fills the
      diagonal with 1.0 (a fidelity assumption), so :meth:`compute_similarities`
      is overridden here to bypass it.
    * Feature entries :math:`v_m(x)_S` are signed Pfaffians and can be
      negative; off-diagonal kernel values are therefore not constrained to be
      nonnegative. The Gram matrix is still PSD (it is a real inner product).

    :param sector_m: Even Majorana sector :math:`m`. Must satisfy
        :math:`2 \\leq m \\leq 2n-2` and be even.
    :param observable_norm: Multiplicative scale :math:`c` on the kernel.
    """

    DEFAULT_SECTOR_M = 2
    DEFAULT_OBSERVABLE_NORM = 1.0

    def __init__(
        self,
        *,
        sector_m: int = DEFAULT_SECTOR_M,
        observable_norm: float = DEFAULT_OBSERVABLE_NORM,
        gram_batch_size: int = NIFKernel.DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = NIFKernel.DEFAULT_RANDOM_STATE,
        alignment: bool = NIFKernel.DEFAULT_ALIGNMENT,
        alignment_iterations: int = NIFKernel.DEFAULT_ALIGNMENT_ITERATIONS,
        alignment_learning_rate: float = NIFKernel.DEFAULT_ALIGNMENT_LEARNING_RATE,
        alignment_early_stopping_patience: int = NIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_PATIENCE,
        alignment_early_stopping_threshold: float = NIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_THRESHOLD,
        n_qubits: int = NIFKernel.DEFAULT_N_QUBITS,
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
        if sector_m % 2 != 0 or sector_m < 2:
            raise ValueError(f"sector_m must be a positive even integer; got {sector_m}.")
        if sector_m > 2 * self.n_qubits - 2:
            raise ValueError(
                f"sector_m={sector_m} exceeds 2n-2={2 * self.n_qubits - 2}; no non-trivial m-subsets exist."
            )
        self._sector_m = sector_m
        self.observable_norm = observable_norm

    def feature_map(self, x: TensorLike) -> torch.Tensor:
        r"""Return the order-:math:`m` Pfaffian feature map.

        :param x: Inputs of shape ``(n_samples, n_features)``.
        :return: Tensor of shape ``(n_samples, n_terms)`` whose row ``i`` is
            :math:`v_m(x_i)`.
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

    def compute_similarities(self, x0: torch.Tensor, x1: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Assemble the Gram matrix as a batched inner product of feature maps.

        Bypasses :class:`~matchcake.ml.kernels.gram_matrix.GramMatrix`, which
        forces the diagonal to 1.0 (a fidelity-kernel assumption that does not
        hold here).

        :param x0: First input batch.
        :param x1: Second input batch.
        :return: Gram matrix of shape ``(len(x0), len(x1))``.
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

    @property
    def n_terms(self) -> int:
        """Dimension of the feature map, :math:`\\binom{2n}{m}`.

        :rtype: int
        """
        return int(self.index_sets.shape[0])

    @property
    def sector_m(self) -> int:
        """Even Majorana sector.

        :rtype: int
        """
        return self._sector_m

    @sector_m.setter
    def sector_m(self, value: int) -> None:
        if value % 2 != 0 or value < 2:
            raise ValueError(f"sector_m must be a positive even integer; got {value}.")
        self._sector_m = value
        self.__dict__.pop("index_sets", None)  # invalidate cached index sets

    @NIFKernel.n_qubits.setter  # type: ignore[attr-defined]
    def n_qubits(self, value: int) -> None:
        NIFKernel.n_qubits.fset(self, value)  # type: ignore[union-attr]
        self.__dict__.pop("index_sets", None)  # invalidate cached index sets

    @cached_property
    def index_sets(self) -> np.ndarray:
        """All :math:`\\binom{2n}{m}` ordered :math:`m`-subsets of Majorana indices.

        :rtype: np.ndarray
        """
        return np.array(
            list(combinations(range(2 * self.n_qubits), self._sector_m)),
            dtype=np.int64,
        )

    @cached_property
    def features_train(self) -> torch.Tensor:
        """Cached Pfaffian feature map of the training set.

        :rtype: torch.Tensor
        """
        return self.feature_map(self.x_train_)

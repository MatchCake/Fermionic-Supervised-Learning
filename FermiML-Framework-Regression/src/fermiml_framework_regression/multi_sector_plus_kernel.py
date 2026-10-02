from functools import cached_property
from itertools import combinations
from typing import List, Sequence, Tuple

import numpy as np
import torch
from matchcake.operations.state_preparation.product_state import ProductState
from matchcake.typing import TensorLike
from matchcake.utils import sector_pfaffian_features
from matchcake.utils.torch_utils import to_tensor

from .nif_covariance_kernel import NIFCovarianceKernel


class MultiSectorPlusCovarianceKernel(NIFCovarianceKernel):
    r"""Span several Majorana orders at once, on the :math:`\ket{+}^{\otimes n}` product state.

    Placed ahead of a concrete
    :class:`~fermiml_framework_regression.nif_covariance_kernel.NIFCovarianceKernel` subclass in
    the bases of a kernel, it makes two changes to that kernel. It defines no ansatz of its own,
    so it is never instantiated directly: the ansatz comes from whichever sibling supplies it.

    First, the circuit starts from :math:`\ket{+}^{\otimes n}` rather than the fermionic vacuum.
    That is what makes the displacement vector :math:`d_\mu = \langle c_\mu \rangle` nonzero, and
    :math:`d_\mu` is the only thing that makes odd Majorana orders carry information at all: on a
    computational basis state every odd-order feature vanishes identically.

    Second, the feature map reads the parity-extended :math:`(2n+1)` covariance matrix and
    concatenates one block per spanned order, so that

    .. math::

        v(x) = \bigl[\, v_{m_1}(x) \;\big|\; v_{m_2}(x) \;\big|\; \cdots \,\bigr],
        \qquad
        \dim v = \sum_k \binom{2n}{m_k}.

    An odd-order subset spans an odd-dimensional skew-symmetric submatrix, whose Pfaffian is
    undefined and whose determinant vanishes identically, so it is closed with the parity index
    :math:`2n` to restore even dimension.

    Orders are evaluated one block at a time rather than as one stacked array, because an
    extended odd subset is one index longer than an even subset of the same order.

    :param sectors: Majorana orders to span, e.g. ``(1, 2, 3)``. Orders must be at least 1;
        an even order is bounded by :math:`2n-2` and an odd one, closed with the parity index,
        by :math:`2n`.
    """

    DEFAULT_SECTORS = (1, 2, 3)

    def __init__(self, *, sectors: Sequence[int] = DEFAULT_SECTORS, **kwargs):
        # The parent validates sector_m as even and bounded by 2n-2, which an odd order would
        # fail, so it is constructed at the default and the spanned orders installed afterwards.
        kwargs.pop("sector_m", None)
        super().__init__(sector_m=2, **kwargs)
        # Stored verbatim rather than normalised: sklearn's clone() compares constructor
        # arguments by identity and GaussianProcessRegressor.fit clones its kernel, so
        # rebuilding the sequence here makes the whole model unfittable. Normalisation lives
        # in :attr:`orders`.
        self.sectors = sectors
        if not self.orders or min(self.orders) < 1:
            raise ValueError(f"Majorana orders must be non-empty and >= 1; got {tuple(self.sectors)}.")
        # The guards the parent applies elsewhere key on _sector_m, so it carries the widest
        # order asked for.
        self._sector_m = max(self.orders)

    def ansatz(self, x: TensorLike):
        r"""Prepend the :math:`\ket{+}^{\otimes n}` product state to the base ansatz.

        :param x: Input batch of shape ``(batch_size, n_features)``.
        :return: Generator yielding the state preparation followed by the base ansatz.
        """
        plus_state = np.tile(np.array([1.0, 1.0]) / np.sqrt(2), (self.n_qubits, 1))
        yield ProductState(plus_state, wires=self.wires)
        yield from super().ansatz(x)

    def feature_map(self, x: TensorLike) -> torch.Tensor:
        """Pfaffian features over every spanned order, from the parity-extended covariance matrix.

        :param x: Inputs of shape ``(n_samples, n_features)``.
        :return: Tensor of shape ``(n_samples, n_terms)``.
        :rtype: torch.Tensor
        """
        x = to_tensor(x, dtype=self.R_DTYPE, device=self.device)  # type: ignore
        features = []
        for start in range(0, len(x), self.gram_batch_size):
            batch = x[start : start + self.gram_batch_size]
            self._q_device.execute_generator(self.ansatz(batch), reset=True)
            extended = self._q_device.extended_covariance_matrix  # (B, 2n+1, 2n+1)
            per_order = [
                to_tensor(sector_pfaffian_features(extended, indices), dtype=self.R_DTYPE, device=self.device)  # type: ignore
                for indices in self.index_sets_per_order
            ]
            features.append(torch.cat(per_order, dim=-1))
        return torch.cat(features, dim=0)

    @property
    def orders(self) -> Tuple[int, ...]:
        """The Majorana orders spanned, sorted and de-duplicated.

        :rtype: Tuple[int, ...]
        """
        return tuple(sorted({int(order) for order in self.sectors}))

    @property
    def parity_index(self) -> int:
        """Index of the parity mode in the extended :math:`(2n+1)`-Majorana algebra.

        :rtype: int
        """
        return 2 * self.n_qubits

    @property
    def n_terms(self) -> int:
        """Total feature dimension, summed over every spanned order.

        :rtype: int
        """
        return int(sum(array.shape[0] for array in self.index_sets_per_order))

    @cached_property
    def index_sets_per_order(self) -> List[np.ndarray]:
        """One index array per spanned order, odd orders closed with the parity index.

        :return: One ``(C(2n, order), order or order + 1)`` array per spanned order.
        :rtype: List[np.ndarray]
        """
        arrays = []
        for order in self.orders:
            bound = 2 * self.n_qubits if order % 2 else 2 * self.n_qubits - 2
            if order > bound:
                raise ValueError(
                    f"Majorana order {order} exceeds {bound} for n={self.n_qubits}; no non-trivial subsets exist."
                )
            subsets = list(combinations(range(2 * self.n_qubits), order))
            if order % 2:
                subsets = [subset + (self.parity_index,) for subset in subsets]
            arrays.append(np.array(subsets, dtype=np.int64))
        return arrays

from functools import cached_property
from itertools import combinations
from math import comb
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from matchcake.operations.state_preparation.product_state import ProductState
from matchcake.typing import TensorLike
from matchcake.utils import sector_pfaffian_features
from matchcake.utils.torch_utils import to_tensor

from .fermi_vanilla_mpf_kernel_classifier import FermiVanillaMpfKernelClassifier


class FermiVanillaMpfPlusKernelClassifier(FermiVanillaMpfKernelClassifier):
    """Order-m Pfaffian covariance kernel classifier initialised in the :math:`|+\\rangle` state.

    Identical to :class:`FermiVanillaMpfKernelClassifier` except that the
    matchgate circuit starts from the product state
    :math:`|+\\rangle^{\\otimes n}` instead of the fermionic vacuum
    :math:`|0\\rangle^{\\otimes n}`. The :math:`|+\\rangle` initial state is
    prepared with :class:`~matchcake.operations.state_preparation.product_state.ProductState`,
    whose Majorana covariance matrix :math:`\\Lambda_0` then propagates through the
    single-particle transition matrix to the order-m Pfaffian feature map.

    This mirrors ``fermiqaoa_maxcut.models.sk_vanilla_plus.SkVanillaPlusModel``,
    which prepends ``ProductState(|+\\rangle)`` before the trainable matchgate
    layer.

    Passing ``sectors`` selects a feature map spanning several Majorana orders at once, computed through the
    parity-extended algebra so that odd orders are admissible. See :meth:`feature_map`. Leaving it unset keeps
    the inherited single-order behaviour exactly, down to the code path taken.

    :param sectors: Majorana orders to span, e.g. ``[2, 3]``. ``None`` uses the single ``sector_m`` order.
    """

    MODEL_NAME = "VanillaMpf+"

    def __init__(self, *args, sectors: Optional[Sequence[int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._sectors = None if sectors is None else tuple(sorted(int(order) for order in sectors))
        if self._sectors is not None:
            if not self._sectors:
                raise ValueError("sectors must name at least one Majorana order.")
            if min(self._sectors) < 1:
                raise ValueError(f"Majorana orders must be >= 1; got {self._sectors}.")
            if len(set(self._sectors)) != len(self._sectors):
                raise ValueError(f"Majorana orders must be distinct; got {self._sectors}.")
            # The guards the parent applies in fit key on _sector_m, so it carries the widest order asked for.
            self._sector_m = max(self._sectors)

    def ansatz(self, x: TensorLike):
        """Prepend the :math:`|+\\rangle^{\\otimes n}` product state to the linear ansatz.

        :param x: Input batch of shape ``(batch_size, n_features)``.
        :return: Generator yielding the product-state preparation followed by the
            single-particle transition matrix operation of the parent ansatz.
        """
        plus_state = np.tile(np.array([1.0, 1.0]) / np.sqrt(2), (self.n_qubits, 1))
        yield ProductState(plus_state, wires=self.wires)
        yield from super().ansatz(x)

    def feature_map(self, x: TensorLike) -> torch.Tensor:
        """Compute the Pfaffian feature map over every spanned Majorana order.

        With a single order this defers to the parent, which reads the plain :math:`2n` covariance matrix.
        With ``sectors`` set it reads the extended :math:`(2n+1)` matrix, whose extra row and column hold the
        displacement vector :math:`d_\\mu = \\langle c_\\mu \\rangle`. That vector is what makes odd orders
        nonzero, and it is nonzero only away from a computational basis state, which is why this lives on the
        :math:`|+\\rangle` variant and not on its vacuum parent.

        Orders are evaluated separately rather than in one stacked array because an extended odd subset is one
        index longer than an even subset of the same order.

        :param x: Input tensor of shape ``(n_samples, n_features)``.
        :return: Tensor of shape ``(n_samples, n_terms)``.
        :rtype: torch.Tensor
        """
        if not self.uses_extended_algebra:
            return super().feature_map(x)
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

    def fit(self, x_train: TensorLike, y_train: Optional[TensorLike] = None):
        """Fit the classifier, validating every spanned order against the qubit count.

        The parent bounds a single even order by :math:`2n-2`. An odd order closed with the parity index only
        needs :math:`m \\le 2n`, so applying the even bound to it would reject feasible settings, notably every
        order-three point on the two-dimensional configurations.

        :param x_train: Training features of shape ``(n_samples, n_features)``.
        :param y_train: Training labels of shape ``(n_samples,)``.
        :return: The fitted classifier.
        :rtype: FermiVanillaMpfPlusKernelClassifier
        """
        if not self.uses_extended_algebra:
            return super().fit(x_train, y_train)
        if self._n_qubits_is_default_flag:
            n_features = np.shape(x_train)[-1]
            # matchgate circuits require an even number of wires
            self.n_qubits = n_features if n_features % 2 == 0 else n_features + 1
        for order in self.sectors:
            bound = 2 * self.n_qubits if order % 2 else 2 * self.n_qubits - 2
            if order > bound:
                raise ValueError(
                    f"Majorana order {order} exceeds {bound} for n={self.n_qubits}; no non-trivial subsets exist."
                )
        n_terms = sum(comb(2 * self.n_qubits, order) for order in self.sectors)
        if n_terms > self.MAX_PFAFFIAN_FEATURE_TERMS:
            raise MemoryError(
                f"Pfaffian feature dimension {n_terms:,} over orders {self.sectors} "
                f"exceeds limit {self.MAX_PFAFFIAN_FEATURE_TERMS:,}. Reduce n_qubits or the orders."
            )
        # Skip the parent's single-order guards, which have already been generalised above.
        return super(FermiVanillaMpfKernelClassifier, self).fit(x_train, y_train)

    @property
    def sectors(self) -> Tuple[int, ...]:
        """The Majorana orders this feature map spans."""
        return self._sectors if self._sectors is not None else (self._sector_m,)

    @property
    def uses_extended_algebra(self) -> bool:
        """Whether the feature map reads the parity-extended covariance matrix."""
        return self._sectors is not None

    @property
    def parity_index(self) -> int:
        """Index of the parity mode in the extended :math:`(2n+1)`-Majorana algebra."""
        return 2 * self.n_qubits

    @property
    def n_terms(self) -> int:
        """Total feature dimension summed over every spanned order."""
        if not self.uses_extended_algebra:
            return int(self.index_sets.shape[0])
        return int(sum(array.shape[0] for array in self.index_sets_per_order))

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        return len(self._q_device.wires)

    @n_qubits.setter
    def n_qubits(self, value: int) -> None:
        # The parent setter invalidates index_sets; the per-order arrays are keyed on n_qubits just as much.
        FermiVanillaMpfKernelClassifier.n_qubits.fset(self, value)
        self.__dict__.pop("index_sets_per_order", None)

    @cached_property
    def index_sets_per_order(self) -> List[np.ndarray]:
        """One index array per spanned order, each closed with the parity index when the order is odd.

        An odd-order Majorana subset has an odd-dimensional skew-symmetric submatrix, whose Pfaffian is
        undefined and whose determinant vanishes identically. Appending the parity index restores even
        dimension, which is what makes the odd orders carry information at all.

        :return: one ``(C(2n, order), order or order + 1)`` array per spanned order.
        :rtype: List[np.ndarray]
        """
        arrays = []
        for order in self.sectors:
            subsets = list(combinations(range(2 * self.n_qubits), order))
            if order % 2:
                subsets = [subset + (self.parity_index,) for subset in subsets]
            arrays.append(np.array(subsets, dtype=np.int64))
        return arrays

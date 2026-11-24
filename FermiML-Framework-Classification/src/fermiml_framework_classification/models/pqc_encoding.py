from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass(frozen=True, eq=False)
class PQCEncoding:
    """Data encoding and circuit layout shared by the PQC and fPQC kernels of https://arxiv.org/abs/2404.19032.

    :param n_features: The number of input features :math:`\\chi`.
    :param n_qubits: The number of qubits :math:`N`.
    :param depth: The number of layers :math:`d`.
    :param reference_angles: The reference angles :math:`\\theta_r`, one per feature.
    """

    REFERENCE_SCALING: ClassVar[float] = np.pi / 2
    DATA_SCALING: ClassVar[float] = np.pi / 2
    # The range the paper's experiments scale features to before Eq. (7).
    FEATURE_RANGE: ClassVar[Tuple[float, float]] = (0.0, 1.0)
    # Qubit-count rules, as divisors of the feature count, from the largest qubit count to the smallest.
    QUBIT_RULES: ClassVar[Dict[str, int]] = {"full": 1, "half": 2, "quarter": 4, "eighth": 8}

    n_features: int
    n_qubits: int
    depth: int
    reference_angles: np.ndarray

    @staticmethod
    def resolve_n_qubits(n_qubits: Optional[Union[int, str]], n_features: int) -> int:
        """Turn a requested qubit count into the one the circuit uses.

        :param n_qubits: A rule, an explicit even number of qubits, or a negative number.
        :param n_features: The number of input features.
        :return: The number of qubits to use.
        :rtype: int
        """
        if n_features < 1:
            raise ValueError(f"n_features must be positive; got {n_features}.")
        largest = max(2, n_features + n_features % 2)
        if n_qubits is None:
            return largest
        if isinstance(n_qubits, str):
            if n_qubits not in PQCEncoding.QUBIT_RULES:
                raise ValueError(f"n_qubits rule must be one of {sorted(PQCEncoding.QUBIT_RULES)}; got {n_qubits!r}.")
            divided = int(np.ceil(n_features / PQCEncoding.QUBIT_RULES[n_qubits]))
            return max(2, divided + divided % 2)
        if n_qubits < 0:
            return largest
        if n_qubits < 2 or n_qubits % 2 != 0:
            raise ValueError(f"n_qubits must be even and at least 2, or negative to infer it; got {n_qubits}.")
        return min(n_qubits, largest)

    @classmethod
    def from_features(cls, n_features: int, n_qubits: Optional[Union[int, str]], random_state: int) -> "PQCEncoding":
        """Build the encoding for a dataset.

        :param n_features: The number of input features.
        :param n_qubits: The requested number of qubits, see :meth:`resolve_n_qubits`.
        :param random_state: The seed of the reference angles. The same seed gives both models the same angles.
        :return: The encoding.
        :rtype: PQCEncoding
        """
        resolved = cls.resolve_n_qubits(n_qubits, n_features)
        reference_angles = np.random.default_rng(random_state).uniform(0.0, 1.0, size=n_features)
        return cls(
            n_features=n_features,
            n_qubits=resolved,
            depth=int(np.ceil(n_features / resolved)),
            reference_angles=reference_angles,
        )

    def angles(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Eq. (7) to scaled inputs.

        :param x: Scaled inputs of shape ``(..., n_features)``.
        :return: The angles, of the same shape, dtype and device as ``x``.
        :rtype: torch.Tensor
        """
        reference = torch.as_tensor(self.reference_angles, dtype=x.dtype, device=x.device)
        return self.REFERENCE_SCALING * reference + self.DATA_SCALING * x

    def layer_angles(self, angles: torch.Tensor, layer: int) -> torch.Tensor:
        """Select the angles a layer loads. The last layer holds fewer than ``n_qubits`` when they do not divide evenly.

        :param angles: Angles of shape ``(..., n_features)``.
        :param layer: The layer index.
        :return: The angles of that layer, of shape ``(..., n_layer_features)``.
        :rtype: torch.Tensor
        """
        return angles[..., layer * self.n_qubits : (layer + 1) * self.n_qubits]

    def entangler_pairs(self, layer: int) -> List[Tuple[int, int]]:
        """List the wire pairs a layer entangles.

        :param layer: The layer index.
        :return: Adjacent wire pairs, starting from wire 0 on even layers and from wire 1 on odd layers.
        :rtype: List[Tuple[int, int]]
        """
        return [(wire, wire + 1) for wire in range(layer % 2, self.n_qubits - 1, 2)]

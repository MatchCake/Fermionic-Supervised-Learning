from typing import ClassVar, Iterator, Optional, Union

import pennylane as qml
import torch
from matchcake.operations import SptmAngleEmbedding, SptmCompZX
from matchcake.typing import TensorLike
from pennylane.operation import Operation

from .nif_kernel_classifier import NIFKernelClassifier
from .pqc_encoding import PQCEncoding


class FermiPQCKernelClassifier(NIFKernelClassifier):
    """Matchgate parameterized quantum circuit fidelity kernel with an SVM head.

    Paper Reference: https://arxiv.org/abs/2404.19032

    :param gram_batch_size: The batch size of the kernel evaluation.
    :param random_state: The seed of the reference angles.
    :param n_qubits: The number of qubits or a rule (``"full"``, ``"half"``, ``"quarter"``, ``"eighth"``), see
        :meth:`PQCEncoding.resolve_n_qubits`, or ``"double"`` for one wire pair per feature.
    :param svm_c: The regularization strength of the SVM.
    """

    MODEL_NAME = "fPQC"
    ROTATIONS = "Y,Z"
    DOUBLE_RULE: ClassVar[str] = "double"

    def __init__(
        self,
        *,
        gram_batch_size: int = NIFKernelClassifier.DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = NIFKernelClassifier.DEFAULT_RANDOM_STATE,
        n_qubits: Union[int, str] = -1,
        svm_c: float = NIFKernelClassifier.DEFAULT_SVM_C,
    ):
        super().__init__(
            gram_batch_size=gram_batch_size,
            random_state=random_state,
            alignment=False,
            svm_c=svm_c,
            n_qubits=2,
        )
        self.requested_n_qubits = n_qubits
        self.encoding_: Optional[PQCEncoding] = None
        self.scaler.feature_range = PQCEncoding.FEATURE_RANGE

    def fit(self, x_train: TensorLike, y_train: Optional[TensorLike] = None) -> "FermiPQCKernelClassifier":
        """Build the encoding for this training set, then fit the kernel and the SVM.

        :param x_train: Training features of shape ``(n_samples, n_features)``.
        :param y_train: Training labels of shape ``(n_samples,)``.
        :return: The fitted classifier.
        :rtype: FermiPQCKernelClassifier
        """
        n_features = int(qml.math.shape(x_train)[-1])
        requested = self.requested_n_qubits
        if isinstance(requested, str) and not self.is_doubled and requested not in PQCEncoding.QUBIT_RULES:
            rules = sorted([*PQCEncoding.QUBIT_RULES, self.DOUBLE_RULE])
            raise ValueError(f"n_qubits rule must be one of {rules}; got {requested!r}.")
        if self.is_doubled:
            # The single-layer layout of the unrestricted model, so both read the same reference angles.
            self.encoding_ = PQCEncoding.from_features(n_features, "full", self.random_state)
            self.n_qubits = 2 * n_features
        else:
            self.encoding_ = PQCEncoding.from_features(n_features, self.requested_n_qubits, self.random_state)
            self.n_qubits = self.encoding_.n_qubits
        return super().fit(x_train, y_train)

    def ansatz(self, x: torch.Tensor) -> Iterator[Operation]:
        """Yield the circuit's operations.

        :param x: Scaled inputs of shape ``(batch_size, n_features)``.
        :return: Generator over the single-particle transition matrices of the circuit.
        :rtype: Iterator[Operation]
        """
        angles = self.encoding_.angles(x)
        if self.is_doubled:
            # Feature j on the even block of pair j, zero on its odd block, which the vacuum never fills. The
            # entanglers of the unrestricted circuit come last and cancel in the fidelity, so none are applied.
            paired = qml.math.reshape(
                qml.math.stack([angles, qml.math.zeros_like(angles)], axis=-1), (-1, self.n_qubits)
            )
            yield from SptmAngleEmbedding(paired, wires=self.wires, rotations=self.ROTATIONS).decomposition()
            return
        for layer in range(self.encoding_.depth):
            layer_angles = self.encoding_.layer_angles(angles, layer)
            yield from SptmAngleEmbedding(layer_angles, wires=self.wires, rotations=self.ROTATIONS).decomposition()
            for first, second in self.encoding_.entangler_pairs(layer):
                yield SptmCompZX(wires=[self.wires[first], self.wires[second]])

    @property
    def is_doubled(self) -> bool:
        """Whether the model runs the doubled encoding, one wire pair per feature.

        :return: ``True`` when ``n_qubits`` is the ``"double"`` rule.
        :rtype: bool
        """
        return self.requested_n_qubits == self.DOUBLE_RULE

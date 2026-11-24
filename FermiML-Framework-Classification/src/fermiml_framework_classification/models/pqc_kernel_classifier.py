from functools import cached_property
from typing import Optional, Union

import pennylane as qml
import torch
from matchcake.typing import TensorLike
from pennylane.measurements import StateMP

from .pqc_encoding import PQCEncoding
from .svs_kernel import SVSKernel


class PQCKernelClassifier(SVSKernel):
    """Unrestricted parameterized quantum circuit fidelity kernel with an SVM head.

    Paper Reference: https://arxiv.org/abs/2404.19032
    Ansatz from: the YZ-CX circuit of https://doi.org/10.1088/2632-2153/acb0b4
    Code Reference: ``PQCKernel`` in https://github.com/MatchCake/kernel_classification

    :param gram_batch_size: The number of states simulated at once. Each costs :math:`2^N` amplitudes.
    :param random_state: The seed of the reference angles and of the device.
    :param n_qubits: The number of qubits or a rule (``"full"``, ``"half"``, ``"quarter"``, ``"eighth"``), see :meth:`PQCEncoding.resolve_n_qubits`.
    :param svm_c: The regularization strength of the SVM.
    """

    MODEL_NAME = "PQC"
    DEFAULT_GRAM_BATCH_SIZE = 32
    ROTATIONS = ("Y", "Z")

    def __init__(
        self,
        *,
        gram_batch_size: int = DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = SVSKernel.DEFAULT_RANDOM_STATE,
        n_qubits: Union[int, str] = -1,
        svm_c: float = SVSKernel.DEFAULT_SVM_C,
    ):
        super().__init__(
            gram_batch_size=gram_batch_size,
            random_state=random_state,
            alignment=False,
            svm_c=svm_c,
        )
        self.requested_n_qubits = n_qubits
        self.encoding_: Optional[PQCEncoding] = None
        self.scaler.feature_range = PQCEncoding.FEATURE_RANGE

    def fit(self, x_train: TensorLike, y_train: Optional[TensorLike] = None) -> "PQCKernelClassifier":
        """Build the encoding for this training set, then fit the kernel and the SVM.

        :param x_train: Training features of shape ``(n_samples, n_features)``.
        :param y_train: Training labels of shape ``(n_samples,)``.
        :return: The fitted classifier.
        :rtype: PQCKernelClassifier
        """
        self.encoding_ = PQCEncoding.from_features(
            int(qml.math.shape(x_train)[-1]), self.requested_n_qubits, self.random_state
        )
        return super().fit(x_train, y_train)

    def ansatz(self, x: torch.Tensor) -> StateMP:
        """Prepare the encoded state.

        :param x: Scaled inputs of shape ``(batch_size, n_features)``.
        :return: The state vector measurement.
        :rtype: StateMP
        """
        # SVSKernel hands the batch over as complex numbers, and these are real angles.
        angles = self.encoding_.angles(qml.math.real(x))
        for layer in range(self.encoding_.depth):
            layer_angles = self.encoding_.layer_angles(angles, layer)
            for rotation in self.ROTATIONS:
                qml.AngleEmbedding(layer_angles, wires=self.wires, rotation=rotation)
            for first, second in self.encoding_.entangler_pairs(layer):
                qml.CNOT(wires=[self.wires[first], self.wires[second]])
        return qml.state()

    @cached_property
    def q_device(self) -> qml.devices.Device:
        return qml.device(self.SIMULATOR, wires=self.encoding_.n_qubits, seed=self.random_state)

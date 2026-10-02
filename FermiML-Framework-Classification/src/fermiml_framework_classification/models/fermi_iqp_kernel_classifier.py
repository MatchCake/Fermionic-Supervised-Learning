from typing import Optional

import numpy as np
import pennylane as qml
import torch
from matchcake.operations import SptmCompHH, SptmCompRzRz, SptmCompZX
from matchcake.typing import TensorLike

from .nif_kernel_classifier import NIFKernelClassifier


class FermiIQPKernelClassifier(NIFKernelClassifier):
    """
    Paper Reference: https://arxiv.org/abs/2403.07059
    Fermionic counterpart of: https://www.nature.com/articles/s41586-019-0980-2
    Code Reference: https://github.com/XanaduAI/qml-benchmarks/blob/main/src/qml_benchmarks/models/iqp_kernel.py
    """

    MODEL_NAME = "fIQP"
    DEFAULT_N_REPEATS = 1

    def __init__(
        self,
        *,
        gram_batch_size: int = NIFKernelClassifier.DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = NIFKernelClassifier.DEFAULT_RANDOM_STATE,
        repeats: int = DEFAULT_N_REPEATS,
        svm_c: float = NIFKernelClassifier.DEFAULT_SVM_C,
    ):
        super().__init__(
            gram_batch_size=gram_batch_size,
            random_state=random_state,
            alignment=False,
            svm_c=svm_c,
            n_qubits=2,
        )
        self.repeats = repeats
        self.scaler.feature_range = (-np.pi / 2, np.pi / 2)

    def fit(self, x_train: TensorLike, y_train: Optional[TensorLike] = None):
        self.n_qubits = qml.math.shape(x_train)[-1]
        return super().fit(x_train, y_train)

    def ansatz(self, x: torch.Tensor):
        wires_indices = [(i, i + 1) for j in range(1, self.n_qubits) for i in range(0, self.n_qubits - j)]
        weights_indices = [(j - 1, i + j) for j in range(1, self.n_qubits) for i in range(0, self.n_qubits - j)]

        for _ in range(self.repeats):
            for i in range(0, self.n_qubits - 1, 2):
                yield SptmCompHH(wires=[self.wires[i], self.wires[i + 1]])
            for i in range(0, self.n_qubits - 1, 2):
                yield SptmCompRzRz(
                    torch.stack([x[..., i], x[..., i + 1]], dim=-1),
                    wires=[self.wires[i], self.wires[i + 1]],
                )
            for (i, j), (k, l) in zip(weights_indices, wires_indices):
                theta = x[..., i] * x[..., j]
                yield SptmCompZX(wires=[self.wires[k], self.wires[l]])
                yield SptmCompRzRz(torch.stack([theta, theta], dim=-1), wires=[self.wires[k], self.wires[l]])
        return

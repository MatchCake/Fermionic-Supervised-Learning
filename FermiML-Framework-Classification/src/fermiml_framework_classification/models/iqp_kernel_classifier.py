from functools import cached_property

import numpy as np
import pennylane as qml
import torch

from .svs_kernel import SVSKernel


class IQPKernelClassifier(SVSKernel):
    """
    Paper Reference: https://arxiv.org/abs/2403.07059
    Model from: https://www.nature.com/articles/s41586-019-0980-2
    Code Reference: https://github.com/XanaduAI/qml-benchmarks/blob/main/src/qml_benchmarks/models/iqp_kernel.py
    """

    MODEL_NAME = "IQP"
    DEFAULT_N_REPEATS = 1

    def __init__(
        self,
        *,
        random_state: int = SVSKernel.DEFAULT_RANDOM_STATE,
        repeats: int = DEFAULT_N_REPEATS,
        svm_c: float = SVSKernel.DEFAULT_SVM_C,
    ):
        super().__init__(
            gram_batch_size=1,
            random_state=random_state,
            alignment=False,
            svm_c=svm_c,
        )
        self.repeats = repeats
        self.scaler.feature_range = (-np.pi / 2, np.pi / 2)

    def ansatz(self, x: torch.Tensor):
        x = qml.math.reshape(x, (-1,))
        assert len(x) == self.n_qubits, f"Input dimension {len(x)} does not match number of qubits {self.n_qubits}"
        qml.IQPEmbedding(x, wires=range(self.n_qubits), n_repeats=self.repeats)
        return qml.state()

    @cached_property
    def q_device(self):
        return qml.device(self.SIMULATOR, wires=qml.math.shape(self.x_train_)[-1], seed=self.random_state)

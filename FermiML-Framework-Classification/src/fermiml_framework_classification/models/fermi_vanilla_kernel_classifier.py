from typing import Optional

import numpy as np
import pennylane as qml
import torch
from matchcake.ml import LinearNIFKernel
from matchcake.operations import SptmCompHH, SptmCompRzRz, SptmCompZX
from matchcake.typing import TensorLike

from .nif_kernel_classifier import NIFKernelClassifier


class FermiVanillaKernelClassifier(NIFKernelClassifier, LinearNIFKernel):
    """
    Fermionic Linear Classifier.
    """

    MODEL_NAME = "Vanilla"

    def __init__(
        self,
        *,
        gram_batch_size: int = NIFKernelClassifier.DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = NIFKernelClassifier.DEFAULT_RANDOM_STATE,
        svm_c: float = NIFKernelClassifier.DEFAULT_SVM_C,
        n_qubits: int = -1,
        **kwargs,
    ):
        self._n_qubits_is_default_flag = False
        if n_qubits is None or n_qubits < 0:
            self._n_qubits_is_default_flag = True
            n_qubits = 2
        super().__init__(
            gram_batch_size=gram_batch_size, random_state=random_state, svm_c=svm_c, n_qubits=n_qubits, **kwargs
        )

    def fit(self, x_train: TensorLike, y_train: Optional[TensorLike] = None):
        if self._n_qubits_is_default_flag:
            self.n_qubits = qml.math.shape(x_train)[-1]
        return super().fit(x_train, y_train)

from functools import cached_property, partial
from typing import Optional, Union

import numpy as np
import pennylane as qml
import torch
from matchcake.ml.kernels.gram_matrix import GramMatrix
from matchcake.ml.kernels.nif_kernel import NIFKernel
from matchcake.typing import TensorLike
from matchcake.utils.torch_utils import to_numpy, to_tensor
from numpy._typing import NDArray
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from tqdm import tqdm


class NIFKernelClassifier(NIFKernel, ClassifierMixin):
    """
    State Vector Simulated Kernel
    """

    MODEL_NAME = "SVSKernel"
    DEFAULT_SVM_C = 1.0

    def __init__(
        self,
        *,
        gram_batch_size: int = NIFKernel.DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = NIFKernel.DEFAULT_RANDOM_STATE,
        alignment: bool = NIFKernel.DEFAULT_ALIGNMENT,
        alignment_iterations: int = NIFKernel.DEFAULT_ALIGNMENT_ITERATIONS,
        alignment_learning_rate: float = NIFKernel.DEFAULT_ALIGNMENT_LEARNING_RATE,
        alignment_early_stopping_patience: int = NIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_PATIENCE,
        alignment_early_stopping_threshold: float = NIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_THRESHOLD,
        svm_c: float = DEFAULT_SVM_C,
        n_qubits: int = NIFKernel.DEFAULT_N_QUBITS,
    ):
        """
        Initializes the class with specific parameters for quantum device configuration and
        random state.

        :param gram_batch_size: The batch size for the Gram computation.
        :param random_state: Seed for the random number generator to ensure reproducibility.
        :param alignment: A boolean flag indicating whether to perform kernel alignment during fitting.
        :param alignment_iterations: The maximum number of iterations for kernel alignment optimization.
        :param alignment_learning_rate: The learning rate for the optimizer used in kernel alignment.
        :param alignment_early_stopping_patience: The number of iterations to wait for improvement before stopping kernel alignment optimization.
        :param alignment_early_stopping_threshold: The threshold for determining improvement in kernel alignment optimization, used for early stopping criteria.
        :param n_qubits: The number of qubits for the non-interacting fermionic device.
        """
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
        self.scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        self.svm_c = svm_c
        self.svm = SVC(kernel="precomputed", probability=True, random_state=random_state)

    def forward(self, x0: TensorLike, x1: Optional[TensorLike] = None, **kwargs) -> TensorLike:
        """
        Calculates the similarities between input tensors and returns the kernel output.

        This method takes two tensors, converts them to the specified dtype, and computes
        their similarities using the `compute_similarities` method of the class. If the second
        input tensor is not provided, it defaults to the first tensor. The result is a kernel
        tensor representing the similarities between the inputs.

        :param x0: The first input tensor.
        :param x1: The second input tensor. Defaults to ``x0`` if not provided.
        :return: A tensor representing the computed kernel similarities.
        """
        if x1 is None:
            x1 = x0
        x0, x1 = self.scaler.transform(to_numpy(x0)), self.scaler.transform(to_numpy(x1))
        x0 = to_tensor(x0, dtype=self.R_DTYPE)  # type: ignore
        x1 = to_tensor(x1, dtype=self.R_DTYPE)  # type: ignore
        self.to(dtype=self.R_DTYPE, device=self.device)
        kernel = self.compute_similarities(x0, x1, **kwargs)  # type: ignore
        return kernel

    def fit(self, x_train: TensorLike, y_train: Optional[TensorLike] = None):
        x_train = to_numpy(x_train)
        self.__dict__.pop("sptms_train", None)  # cached from a previous fit
        self.scaler.fit(x_train)
        super().fit(x_train, y_train)
        self.svm.C = self.svm_c
        x = to_numpy(self.transform(self.x_train_))
        self.svm.fit(x, self.y_train_)
        return self

    def predict(self, x: Union[NDArray, TensorLike]) -> Union[NDArray, torch.Tensor]:
        """
        Predict transformed data using the model.

        This function applies a transformation to the input data and returns the
        resulting output. It supports both NumPy arrays and PyTorch tensors as input.
        The return type will match the input type.

        The model is set to evaluation mode during prediction to ensure that layers
        like dropout and batch normalization behave appropriately for inference.
        The transformation is performed without tracking gradients to optimize performance.

        :param x: The input data to be transformed. Can be either a NumPy array
            or a PyTorch tensor.
        :return: The transformed input data. The returned type will match the
            type of the input data (NumPy array or PyTorch tensor).
        """
        self.eval()
        with torch.no_grad():
            x = to_tensor(x, dtype=self.R_DTYPE, device=self.device)
            x = self.transform(x)
            x = to_numpy(x)
            return self.svm.predict(x)

    def predict_proba(self, x: Union[NDArray, TensorLike]) -> Union[NDArray, torch.Tensor]:
        """
        Predict transformed data using the model.

        This function applies a transformation to the input data and returns the
        resulting output. It supports both NumPy arrays and PyTorch tensors as input.
        The return type will match the input type.

        The model is set to evaluation mode during prediction to ensure that layers
        like dropout and batch normalization behave appropriately for inference.
        The transformation is performed without tracking gradients to optimize performance.

        :param x: The input data to be transformed. Can be either a NumPy array
            or a PyTorch tensor.
        :return: The transformed input data. The returned type will match the
            type of the input data (NumPy array or PyTorch tensor).
        """
        self.eval()
        with torch.no_grad():
            x = to_tensor(x, dtype=self.R_DTYPE, device=self.device)
            x = self.transform(x)
            x = to_numpy(x)
            return self.svm.predict_proba(x)

    def compute_similarities(self, x0: torch.Tensor, x1: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the fidelity kernel between ``x0`` and ``x1``.

        :param x0: Scaled inputs of shape ``(n_samples_0, n_features)``.
        :param x1: Scaled inputs of shape ``(n_samples_1, n_features)``.
        :return: The kernel, of shape ``(n_samples_0, n_samples_1)``.
        :rtype: torch.Tensor
        """
        if x0.shape == x1.shape and torch.equal(x0, x1):
            return super().compute_similarities(x0, x1, **kwargs)
        sptm0 = self._x_to_sptm(x0)
        sptm1 = self.sptms_train if kwargs.get("x1_train", False) else self._x_to_sptm(x1)
        n_rows, n_cols = int(sptm0.shape[0]), int(sptm1.shape[0])
        row_indices, col_indices = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
        all_indices = np.stack([row_indices.reshape(-1), col_indices.reshape(-1)], axis=1)
        p_bar = tqdm(
            total=len(all_indices),
            desc="Computing kernel",
            unit="element",
            disable=not kwargs.get("verbose", False),
        )
        values = []
        for start in range(0, len(all_indices), self.gram_batch_size):
            batch_indices = all_indices[start : start + self.gram_batch_size]
            batch_values = self._compute_similarities_func(batch_indices, sptm0, sptm1, p_bar)
            values.append(torch.real(to_tensor(batch_values, device=self.device)).reshape(-1))
        p_bar.close()
        return torch.cat(values).reshape(n_rows, n_cols).to(dtype=self.R_DTYPE)

    @cached_property
    def sptms_train(self) -> TensorLike:
        x = to_tensor(self.scaler.transform(to_numpy(self.x_train_)), dtype=self.R_DTYPE)
        return self._x_to_sptm(x)

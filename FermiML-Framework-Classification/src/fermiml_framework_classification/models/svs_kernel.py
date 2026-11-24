from functools import cached_property
from typing import Optional, Union

import numpy as np
import pennylane as qml
import torch
from matchcake.ml.kernels.kernel import Kernel
from matchcake.typing import TensorLike
from matchcake.utils.torch_utils import to_numpy, to_tensor
from numpy._typing import NDArray
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC


class SVSKernel(Kernel, ClassifierMixin):
    """
    State Vector Simulated Kernel
    """

    MODEL_NAME = "SVSKernel"
    SIMULATOR = "default.qubit"
    DEFAULT_SVM_C = 1.0

    def __init__(
        self,
        *,
        gram_batch_size: int = Kernel.DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = Kernel.DEFAULT_RANDOM_STATE,
        alignment: bool = Kernel.DEFAULT_ALIGNMENT,
        alignment_iterations: int = Kernel.DEFAULT_ALIGNMENT_ITERATIONS,
        alignment_learning_rate: float = Kernel.DEFAULT_ALIGNMENT_LEARNING_RATE,
        alignment_early_stopping_patience: int = Kernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_PATIENCE,
        alignment_early_stopping_threshold: float = Kernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_THRESHOLD,
        svm_c: float = DEFAULT_SVM_C,
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
        )
        self.R_DTYPE = torch.float32
        self.C_DTYPE = torch.complex64
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
        self.invalidate_device_cache()
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

    def ansatz(self, x: torch.Tensor):
        """
        Represents an abstract method for implementing a specific ansatz.

        This method must be overridden in a subclass and is not implemented in
        the base class itself. It serves as a blueprint for any specific ansatz
        that is required by the framework or application.

        The responsibility of providing the implementation for the ansatz logic
        remains with the derived class.

        :param x: Input tensor.
        :type x: torch.Tensor
        :raises NotImplementedError: If the method is called without overriding
            it in a subclass.
        """
        raise NotImplementedError

    def compute_similarities(self, x0: torch.Tensor, x1: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the fidelity kernel between ``x0`` and ``x1`` from the overlaps of their state vectors.

        :param x0: Scaled inputs of shape ``(n_samples_0, n_features)``.
        :param x1: Scaled inputs of shape ``(n_samples_1, n_features)``.
        :return: The kernel, of shape ``(n_samples_0, n_samples_1)``.
        :rtype: torch.Tensor
        """
        # Simulated in the order the previous GramMatrix path used, and never reused: a seeded sampling device (QKS)
        # draws from one stream, so any change in the number or order of simulations changes its results.
        states0 = self._x_to_sv(x0)
        states1 = self.sv_train if kwargs.get("x1_train", False) else self._x_to_sv(x1)
        kernel = (torch.abs(states0.conj() @ states1.T) ** 2).to(dtype=self.R_DTYPE)  # (n_samples_0, n_samples_1)
        if x0.shape == x1.shape and torch.equal(x0, x1):
            kernel = (kernel + kernel.T) / 2
            kernel.fill_diagonal_(1.0)
        return kernel

    def _x_to_sv(self, x: TensorLike) -> TensorLike:
        """
        Transforms the given input tensor into a state vector by
        applying the user's ansatz.

        :param x: Input tensor to be transformed.
        :type x: TensorLike
        :return: The state vector obtained after applying the ansatz.
        :rtype: TensorLike
        """
        x = to_tensor(x, dtype=self.R_DTYPE)  # type: ignore
        batched_indices = np.array_split(np.arange(len(x)), np.ceil(len(x) / self.gram_batch_size))
        sv_list = []
        for batch_indices in batched_indices:
            bx = to_tensor(x[batch_indices], dtype=self.C_DTYPE, device=self.device)  # type: ignore
            new_sv = to_tensor(self.qnode(bx), dtype=self.C_DTYPE, device=x.device).reshape(-1, 2**self.n_qubits)
            sv_list.append(new_sv)
        stacked_sv = qml.math.concatenate(sv_list, axis=0).reshape(x.shape[0], 2**self.n_qubits)
        return stacked_sv

    @property
    def wires(self):
        return self.q_device.wires

    @property
    def n_qubits(self) -> int:
        return len(self.q_device.wires)

    def invalidate_device_cache(self) -> None:
        """Drop the cached device, qnode and training state vectors.

        The device is cached rather than rebuilt on every access because a seeded device carries the sampling
        generator: rebuilding it per call would restart that generator and make every draw identical. The cache
        must therefore be dropped whenever the training set changes, since the wire count is read off it.
        """
        for attribute in ("q_device", "qnode", "sv_train"):
            self.__dict__.pop(attribute, None)

    @cached_property
    def q_device(self):
        return qml.device(self.SIMULATOR, seed=self.random_state)  # pragma: no cover

    @cached_property
    def qnode(self):
        return qml.QNode(self.ansatz, self.q_device)

    @cached_property
    def sv_train(self) -> TensorLike:
        x = to_tensor(self.scaler.transform(to_numpy(self.x_train_)), dtype=self.R_DTYPE)
        return self._x_to_sv(x)

from functools import cached_property
from typing import Optional, Union

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from matchcake.ml.kernels.kernel import Kernel
from matchcake.typing import TensorLike
from matchcake.utils.torch_utils import to_numpy, to_tensor
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import softmax

from .svs_kernel import SVSKernel


class KitchenSinkKernelClassifier(SVSKernel):
    """
    Quantum Kitchen Sinks kernel classifier using standard PennyLane gates.

    Paper Reference: https://arxiv.org/abs/1806.08321
    Code Reference: https://github.com/XanaduAI/qml-benchmarks/blob/main/src/qml_benchmarks/models/quantum_kitchen_sinks.py
    """

    MODEL_NAME = "QKS"
    DEFAULT_N_EPISODES = 100
    DEFAULT_N_QFEATURES = "full"
    DEFAULT_VAR = 1.0

    def __init__(
        self,
        *,
        n_episodes: int = DEFAULT_N_EPISODES,
        n_qfeatures: Union[int, str] = DEFAULT_N_QFEATURES,
        var: float = DEFAULT_VAR,
        random_state: int = SVSKernel.DEFAULT_RANDOM_STATE,
    ):
        super().__init__(
            gram_batch_size=1,
            random_state=random_state,
            alignment=False,
        )
        self.n_episodes = n_episodes
        self.n_qfeatures = n_qfeatures
        self.var = var
        self.weights_: Optional[nn.Parameter] = None
        self.bias_: Optional[nn.Parameter] = None
        self.cls_head = LogisticRegression(penalty=None, solver="lbfgs", tol=10e-4)

    def fit(self, x_train: TensorLike, y_train: Optional[TensorLike] = None):
        self.x_train_ = x_train
        self.invalidate_device_cache()
        n_features = qml.math.shape(self.x_train_)[-1]
        rng = np.random.default_rng(self.random_state)
        self.weights_ = nn.Parameter(
            torch.tensor(
                rng.standard_normal((self.n_episodes, self.n_qubits, n_features)) * np.sqrt(self.var),
                dtype=self.R_DTYPE,
            ),
        )
        self.bias_ = nn.Parameter(
            torch.tensor(
                rng.uniform(0, 2 * np.pi, (self.n_episodes, self.n_qubits)),
                dtype=self.R_DTYPE,
            ),
        )
        self.scaler.fit(to_numpy(x_train))
        Kernel.fit(self, x_train, y_train)
        self.cls_head.fit(to_numpy(self.sv_train), self.y_train_)
        self.classes_ = self.cls_head.classes_
        return self

    def predict(self, x: TensorLike) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            predictions = self.predict_proba(x)
            mapped_predictions = np.argmax(predictions, axis=1)
            return np.take(self.classes_, mapped_predictions)

    def predict_proba(self, x: TensorLike) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x_scaled = self.scaler.transform(to_numpy(x))
            qks_features = to_numpy(self._x_to_sv(to_tensor(x_scaled, dtype=self.R_DTYPE)))
            predictions_2d = self.cls_head.predict_proba(qks_features)
            return softmax(predictions_2d, copy=False)

    def ansatz(self, x: torch.Tensor):
        x = qml.math.reshape(x, (-1,))
        for i, q in enumerate(x):
            qml.RX(q, wires=i)
        for i in range(0, self.n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
        for i in range(0, self.n_qubits - 2):
            qml.CNOT(wires=[i, i + 2])
        return qml.sample(wires=range(self.n_qubits))

    def _x_to_sv(self, x: TensorLike) -> TensorLike:
        x = to_tensor(x, dtype=self.R_DTYPE)  # type: ignore
        x_prime = (
            qml.math.einsum("...f,pnf->...pn", x, self.weights_)  # type: ignore
            + self.bias_.reshape(1, self.n_episodes, self.n_qubits)  # type: ignore
        ).reshape(-1, self.n_episodes, self.n_qubits)
        batched_indices = np.array_split(np.arange(len(x)), np.ceil(len(x) / self.gram_batch_size))
        qks_features = []
        for batch_indices in batched_indices:
            z_list = []
            for k in range(self.n_episodes):
                bx = to_tensor(x_prime[batch_indices, k], dtype=self.C_DTYPE, device=self.device)  # type: ignore
                z = self.qnode(bx)
                new_z = torch.tensor(to_numpy(z), dtype=self.R_DTYPE, device=x.device).reshape(-1, self.n_qubits)
                z_list.append(new_z)
            new_qks_features = torch.cat(z_list, dim=-1)
            qks_features.append(new_qks_features)
        stacked_qks_features = qml.math.concatenate(qks_features, axis=0).reshape(
            x.shape[0], self.n_episodes * self.n_qubits
        )
        return stacked_qks_features

    def _compute_similarities_func(self, indices, sv0, sv1, p_bar):
        b_qks0, b_qks1 = sv0[indices[:, 0]], sv1[indices[:, 1]]
        b_qks0 = to_tensor(b_qks0, dtype=self.R_DTYPE, device=self.device)  # type: ignore
        b_qks1 = to_tensor(b_qks1, dtype=self.R_DTYPE, device=self.device)  # type: ignore
        out = qml.math.einsum("...i,...i->...", b_qks0, b_qks1) / b_qks0.shape[-1]
        p_bar.update(len(indices))
        return out

    @cached_property
    def q_device(self):
        n_features = qml.math.shape(self.x_train_)[-1]
        if self.n_qfeatures == "full":
            n_qubits = n_features
        elif self.n_qfeatures == "half":
            n_qubits = int(np.ceil(n_features / 2))
        else:
            n_qubits = int(self.n_qfeatures)
        # The seed makes the one-shot draws reproducible. It only works because the device is cached: a device
        # rebuilt per episode would restart its generator and hand every episode the same sample.
        return qml.device(self.SIMULATOR, wires=n_qubits, seed=self.random_state)

    @cached_property
    def qnode(self):
        return qml.set_shots(qml.QNode(self.ansatz, self.q_device), shots=1)

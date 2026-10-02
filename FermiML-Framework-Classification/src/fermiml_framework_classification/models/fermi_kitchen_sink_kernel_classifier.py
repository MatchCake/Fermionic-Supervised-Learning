from typing import Optional, Union

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from matchcake import NonInteractingFermionicDevice
from matchcake.ml.kernels.kernel import Kernel
from matchcake.operations import SptmCompRxRx, SptmCompZX
from matchcake.typing import TensorLike
from matchcake.utils.torch_utils import to_numpy, to_tensor
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import softmax

from .nif_kernel_classifier import NIFKernelClassifier
from .svs_kernel import SVSKernel


class FermiKitchenSinkKernelClassifier(NIFKernelClassifier):
    """
    Fermionic counterpart of `KitchenSinkKernelClassifier`
    """

    MODEL_NAME = "fQKS"
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
            n_qubits=2,
        )
        self.n_episodes = n_episodes
        self.n_qfeatures = n_qfeatures
        self.var = var
        self.weights_: Optional[nn.Parameter] = None
        self.bias_: Optional[nn.Parameter] = None
        self.cls_head = LogisticRegression(penalty=None, solver="lbfgs", tol=10e-4)

    def _resolve_n_qubits(self, n_features: int) -> int:
        if self.n_qfeatures == "full":
            n_qubits = n_features
        elif self.n_qfeatures == "half":
            n_qubits = int(np.ceil(n_features / 2))
        else:
            n_qubits = int(self.n_qfeatures)
        return n_qubits + (n_qubits % 2)

    def fit(self, x_train: TensorLike, y_train: Optional[TensorLike] = None):
        n_features = qml.math.shape(x_train)[-1]
        self.n_qubits = self._resolve_n_qubits(n_features)
        self._q_device = NonInteractingFermionicDevice(self.n_qubits, r_dtype=self.R_DTYPE)
        self.__dict__.pop("sptms_train", None)

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
        self.cls_head.fit(to_numpy(self.sptms_train), self.y_train_)
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
            qks_features = to_numpy(self._x_to_sptm(to_tensor(x_scaled, dtype=self.R_DTYPE)))
            predictions_2d = self.cls_head.predict_proba(qks_features)
            return softmax(predictions_2d, copy=False)

    def ansatz(self, x: torch.Tensor):
        for i in range(0, self.n_qubits - 1, 2):
            yield SptmCompRxRx(
                torch.stack([x[..., i], x[..., i + 1]], dim=-1),
                wires=[i, i + 1],
            )
        for i in range(0, self.n_qubits - 1, 2):
            yield SptmCompZX(wires=[i, i + 1])
        for i in range(1, self.n_qubits - 1, 2):
            yield SptmCompZX(wires=[i, i + 1])

    def _x_to_sptm(self, x: TensorLike) -> TensorLike:
        """
        This function doesn't compute the SPTM, but rather the QKS features.

        This will be taken into account in the method `_compute_similarities_func`.
        """
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
                bx = to_tensor(x_prime[batch_indices, k], dtype=self.R_DTYPE, device=self.device)  # type: ignore
                # Shots live on the circuit, not the device (PennyLane 0.45 deprecates
                # device-level shots); one sample per episode yields the QKS measurement.
                tape = qml.tape.QuantumScript(
                    list(self.ansatz(bx)),
                    [qml.sample(wires=range(self._q_device.num_wires))],
                    shots=1,
                )
                z = self._q_device.execute(tape)
                new_z = to_tensor(z, dtype=self.R_DTYPE, device=x.device).reshape(-1, self._q_device.num_wires)
                z_list.append(new_z)
            new_qks_features = torch.cat(z_list, dim=-1)
            qks_features.append(new_qks_features)
        stacked_qks_features = qml.math.concatenate(qks_features, axis=0).reshape(
            x.shape[0], self.n_episodes * self.n_qubits
        )
        return stacked_qks_features

    def _compute_similarities_func(self, indices, sptm0, sptm1, p_bar):
        b_qks0, b_qks1 = sptm0[indices[:, 0]], sptm1[indices[:, 1]]
        b_qks0 = to_tensor(b_qks0, dtype=self.R_DTYPE, device=self.device)  # type: ignore
        b_qks1 = to_tensor(b_qks1, dtype=self.R_DTYPE, device=self.device)  # type: ignore
        out = qml.math.einsum("...i,...i->...", b_qks0, b_qks1) / b_qks0.shape[-1]
        p_bar.update(len(indices))
        return out

    @property
    def n_qubits(self) -> int:
        return len(self.q_device.wires)

    @n_qubits.setter
    def n_qubits(self, value: int):
        self._q_device = NonInteractingFermionicDevice(value, r_dtype=self.R_DTYPE)

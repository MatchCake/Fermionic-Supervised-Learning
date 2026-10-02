import argparse
from functools import cached_property
from typing import Any, List, Optional

import numpy as np
import pennylane as qml
import torch
from ax import ChoiceParameterConfig, RangeParameterConfig
from matchcake import NonInteractingFermionicDevice
from matchcake.operations import (
    SingleParticleTransitionMatrixOperation,
)
from matchcake_opt.modules.classification_model import ClassificationModel


class NIFIsingCls(ClassificationModel):
    MODEL_NAME = "NIFIsingCls"
    DEFAULT_N_QUBITS = 48
    DEFAULT_LEARNING_RATE = 1e-3
    DEFAULT_ENCODER_OUTPUT_ACTIVATION = "Identity"

    HP_CONFIGS: List[Any] = [
        RangeParameterConfig(
            name="learning_rate",
            parameter_type="float",
            bounds=(1e-5, 0.1),
        ),
        RangeParameterConfig(
            name="n_qubits",
            parameter_type="int",
            bounds=(4, 92),
        ),
        ChoiceParameterConfig(
            name="encoder_output_activation",
            parameter_type="str",
            values=["Tanh", "Identity"],
            is_ordered=False,
        ),
    ]

    @classmethod
    def add_model_specific_args(cls, parent_parser: Optional[argparse.ArgumentParser] = None):
        parent_parser = super().add_model_specific_args(parent_parser)
        if parent_parser is None:
            parent_parser = argparse.ArgumentParser()
        parser = parent_parser.add_argument_group(f"{cls.MODEL_NAME} Model")
        parser.add_argument(
            "--n_qubits",
            type=int,
            default=cls.DEFAULT_N_QUBITS,
            help="Number of qubits",
        )
        parser.add_argument(
            "--encoder_output_activation",
            type=str,
            default=cls.DEFAULT_ENCODER_OUTPUT_ACTIVATION,
            choices=["Tanh", "Identity"],
            help="Activation function for the encoder output layer",
        )
        return parent_parser

    def __init__(
        self,
        input_shape: Optional[tuple[int, ...]],
        output_shape: Optional[tuple[int, ...]],
        learning_rate: float = DEFAULT_LEARNING_RATE,
        n_qubits: Optional[int] = DEFAULT_N_QUBITS,
        encoder_output_activation: str = DEFAULT_ENCODER_OUTPUT_ACTIVATION,
        **kwargs,
    ):
        super().__init__(input_shape=input_shape, output_shape=output_shape, learning_rate=learning_rate, **kwargs)
        self.save_hyperparameters(
            "learning_rate",
            "n_qubits",
            "encoder_output_activation",
        )
        if n_qubits is None:
            assert self.input_shape is not None, "input_shape must be provided if n_qubits is None"
            n_qubits = int(np.prod(self.input_shape))  # type: ignore
            print(f"Using inferred n_qubits: {n_qubits} from input size {self.input_shape}")
        self.n_qubits: int = n_qubits
        self.R_DTYPE = torch.float32
        self.C_DTYPE = torch.cfloat
        self.q_device = NonInteractingFermionicDevice(
            wires=self.n_qubits, r_dtype=self.R_DTYPE, c_dtype=self.C_DTYPE, show_progress=False
        )
        self.encoder_output_activation = encoder_output_activation
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.LazyBatchNorm1d(),
            torch.nn.LazyLinear(self.embedding_dim),
            getattr(torch.nn, encoder_output_activation)(),
        )
        self.bias = torch.nn.Parameter(torch.zeros((self.embedding_dim,), dtype=self.R_DTYPE), requires_grad=True)
        self.weights = torch.nn.Parameter(torch.rand((int(self.output_size), self.n_sptm_params)), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.weights)
        self._build()

    def _build(self):
        dummy_input = torch.randn((3, *self.input_shape)).to(device=self.device)
        with torch.no_grad():
            self(dummy_input)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_input(x)
        x = self.encoder(x)
        x = x + self.bias.view(1, -1)
        self.q_device.execute_generator(self.circuit_gen(), reset=True)
        probs = self.q_device.probability(wires=self.observable_batched_wires).to(
            dtype=self.q_device.R_DTYPE, device=self.torch_device
        )
        zz_eigvals = torch.from_numpy(np.asarray([1, -1, -1, 1])).to(dtype=self.R_DTYPE, device=self.device)
        x = torch.einsum("...i,kij,j->...k", x, probs, zz_eigvals)
        return x

    def circuit_gen(self):
        yield qml.BasisState(self.initial_basis_state, wires=self.wires)
        yield self.get_sptm_weights()
        return

    def get_sptm_weights(self):
        """
        Compute the single-particle transition matrix (SPTM) weights based on the initialized weight tensor.

        This method constructs a tensor, `h`, which encodes the pairwise weight interactions
        in a prescribed upper triangular form. It then symmetrizes `h` to ensure anti-symmetry
        about the main diagonal. Afterward, the matrix exponential of `h` is computed to generate
        the SPTM. Finally, the SPTM is encapsulated in a `SingleParticleTransitionMatrixOperation`
        object for further usage.

        :return: An instance of `SingleParticleTransitionMatrixOperation` that encapsulates
            the computed single-particle transition matrix based on the initialized weights.
            The shape of the SPTM is (n_classes, 2 * n_qubits, 2 * n_qubits).
        :rtype: SingleParticleTransitionMatrixOperation
        """
        h = torch.zeros(
            (int(self.output_size), 2 * self.n_qubits, 2 * self.n_qubits), dtype=self.R_DTYPE, device=self.torch_device
        )
        triu_indices = np.triu_indices(2 * self.n_qubits, k=1)
        h[:, triu_indices[0], triu_indices[1]] = self.weights
        h = h - torch.einsum("...ij->...ji", h)
        sptm = torch.matrix_exp(h)
        return SingleParticleTransitionMatrixOperation(sptm, wires=self.wires)

    def preprocess_input(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1, *self.input_shape)  # type: ignore
        # Average over the sequence dimension if it exists, otherwise keep the input as is
        x = torch.mean(x, dim=1)
        return x

    def extra_repr(self) -> str:
        elements = [
            f"n_qubits={self.n_qubits}",
            f"output_size={self.output_size}",
            f"encoder_output_activation={self.encoder_output_activation}",
        ]
        return ", ".join(elements)

    @cached_property
    def observable_batched_wires(self):
        return np.stack(np.triu_indices(self.n_qubits, k=1), axis=-1).tolist()

    @cached_property
    def embedding_dim(self):
        return np.triu_indices(self.n_qubits, k=1)[0].size

    @cached_property
    def n_sptm_params(self):
        return np.triu_indices(2 * self.n_qubits, k=1)[0].size

    @property
    def torch_device(self):
        return self.device

    @property
    def wires(self):
        return self.q_device.wires

    @property
    def initial_basis_state(self):
        return np.zeros(self.n_qubits, dtype=int)

    @property
    def output_size(self):
        return int(np.prod(self.output_shape))

    @property
    def num_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

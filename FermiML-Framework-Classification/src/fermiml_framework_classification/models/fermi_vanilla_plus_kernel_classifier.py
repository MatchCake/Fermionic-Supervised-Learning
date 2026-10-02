import numpy as np
import torch
from matchcake.operations.state_preparation.product_state import ProductState
from matchcake.typing import TensorLike
from matchcake.utils.torch_utils import to_tensor
from tqdm import tqdm

from .fermi_vanilla_kernel_classifier import FermiVanillaKernelClassifier


class FermiVanillaPlusKernelClassifier(FermiVanillaKernelClassifier):
    """Fermionic linear classifier initialised in the :math:`|+\\rangle` state.

    Identical to :class:`FermiVanillaKernelClassifier` except that the kernel is
    evaluated from the product state :math:`|+\\rangle^{\\otimes n}` rather than the
    fermionic vacuum :math:`|0\\rangle^{\\otimes n}`. The :math:`|+\\rangle` state is
    prepared with
    :class:`~matchcake.operations.state_preparation.product_state.ProductState`,
    evolved by the matchgate single-particle transition matrices of both inputs,
    and each entry is the transition probability to the
    :math:`|0\\rangle^{\\otimes n}` basis state:

    .. math::

        M(x_i, x_j) = \\bigl| \\langle 0 |\\, U(x_j)^\\dagger U(x_i)\\, |+\\rangle \\bigr|^2 .

    Because the bra is :math:`|0\\rangle` and the ket is :math:`|+\\rangle`, this matrix
    is **not symmetric** in general (unlike the vacuum fidelity kernel, whose bra
    and ket are both :math:`|0\\rangle`). The full matrix is returned as-is and fed to
    the precomputed-kernel SVM. The constant diagonal equals
    :math:`|\\langle 0 | + \\rangle|^2 = 1/2^n`.

    This mirrors ``fermiqaoa_maxcut.models.sk_vanilla_plus.SkVanillaPlusModel``,
    which prepends ``ProductState(|+\\rangle)`` before the matchgate evolution and
    reads an expectation value via the non-interacting fermionic device.
    """

    MODEL_NAME = "Vanilla+"

    def compute_similarities(self, x0: torch.Tensor, x1: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute the full asymmetric transition-probability matrix between x0 and x1.

        The matchgate fidelity machinery in
        :class:`~matchcake.ml.kernels.nif_kernel.NIFKernel` assembles only the lower
        triangle of a Gram matrix and mirrors it, assuming a symmetric kernel. The
        :math:`|+\\rangle` transition-probability matrix is asymmetric, so every
        ``(i, j)`` entry is computed explicitly here instead.

        :param x0: First input tensor of shape ``(n_samples_0, n_features)``.
        :param x1: Second input tensor of shape ``(n_samples_1, n_features)``.
        :return: Transition-probability matrix of shape ``(n_samples_0, n_samples_1)``.
        :rtype: torch.Tensor
        """
        sptm0 = self._x_to_sptm(x0)
        if kwargs.get("x1_train", False):
            sptm1 = self.sptms_train
        else:
            sptm1 = self._x_to_sptm(x1)
        n_rows = int(sptm0.shape[0])
        n_cols = int(sptm1.shape[0])
        row_indices, col_indices = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
        all_indices = np.stack([row_indices.reshape(-1), col_indices.reshape(-1)], axis=1)
        p_bar = tqdm(
            total=len(all_indices),
            desc="Computing transition-probability matrix",
            unit="element",
            disable=not kwargs.get("verbose", False),
        )
        values = []
        for start in range(0, len(all_indices), self.gram_batch_size):
            batch_indices = all_indices[start : start + self.gram_batch_size]
            batch_values = self._compute_similarities_func(batch_indices, sptm0, sptm1, p_bar)
            batch_tensor = to_tensor(batch_values, dtype=self.R_DTYPE, device=self.device)
            values.append(torch.real(batch_tensor).reshape(-1))
        p_bar.close()
        gram = torch.cat(values).reshape(n_rows, n_cols)
        return gram.to(dtype=self.R_DTYPE)

    def circuit(self, sptm0: TensorLike, sptm1: TensorLike):
        """Prepend the :math:`|+\\rangle^{\\otimes n}` product state to the fidelity circuit.

        :param sptm0: Single-particle transition matrix of the first input.
        :param sptm1: Single-particle transition matrix of the second input, applied
            as its adjoint.
        :return: Generator yielding the product-state preparation followed by the
            single-particle transition matrix operations of the parent circuit.
        """
        plus_state = np.tile(np.array([1.0, 1.0]) / np.sqrt(2), (self.n_qubits, 1))
        yield ProductState(plus_state, wires=self.wires)
        yield from super().circuit(sptm0, sptm1)

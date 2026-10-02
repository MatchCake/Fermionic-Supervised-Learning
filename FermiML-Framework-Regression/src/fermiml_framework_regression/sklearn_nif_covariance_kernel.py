from typing import Optional

import numpy as np
import torch
from sklearn.gaussian_process.kernels import Kernel as _SklearnKernel

from .nif_covariance_kernel import NIFCovarianceKernel


class SklearnNIFCovarianceKernel(_SklearnKernel):
    r"""Wrap a :class:`NIFCovarianceKernel` as a scikit-learn GP kernel.

    :class:`~sklearn.gaussian_process.GaussianProcessRegressor` requires an
    object implementing the sklearn ``Kernel`` interface (``__call__`` and
    ``diag``). This adapter provides both without adding tunable hyperparameters
    -- any tuning of the matchgate circuit happens inside the
    :class:`NIFCovarianceKernel` (e.g. via kernel alignment) before wrapping.

    Two points specific to the covariance kernel:

    * ``diag`` returns the genuine per-point value
      :math:`\tfrac{cs}{2^n}\lVert v_m(x)\rVert^2`, not ones. A
      fidelity-style ``diag`` of ones would be inconsistent with ``__call__``.
    * The kernel is non-stationary, so ``is_stationary`` returns ``False``.

    Practical note: ``GaussianProcessRegressor(normalize_y=True)`` ill-conditions
    this kernel; standardise ``y`` by hand before fitting instead.

    :param kernel: A fitted (and optionally frozen) :class:`NIFCovarianceKernel`
        subclass instance.
    :param scale: Extra multiplicative scale applied on top of the kernel's own
        ``observable_norm`` prefactor.
    """

    def __init__(self, kernel: NIFCovarianceKernel, scale: float = 1.0):
        self.kernel = kernel
        self.scale = scale

    def __call__(self, X: np.ndarray, Y: Optional[np.ndarray] = None, eval_gradient: bool = False):
        """Evaluate the kernel matrix K(X, Y).

        :param X: First input array of shape ``(n_samples_a, n_features)``.
        :param Y: Second input array. If ``None``, use ``X``.
        :param eval_gradient: If ``True``, also return the gradient (empty
            here since there are no sklearn hyperparameters).
        :return: Kernel matrix of shape ``(n_a, n_b)``, and optionally an
            empty gradient array of shape ``(n_a, n_b, 0)``.
        """
        gram = self._gram(X, Y).detach().cpu().numpy()
        if eval_gradient:
            return gram, np.empty((gram.shape[0], gram.shape[1], 0))
        return gram

    def diag(self, X: np.ndarray) -> np.ndarray:
        """Return the diagonal of K(X, X).

        :param X: Input array of shape ``(n_samples, n_features)``.
        :return: Diagonal values of shape ``(n_samples,)``.
        """
        v0 = self.kernel.feature_map(X)
        prefactor = self.scale * self.kernel.observable_norm / (2**self.kernel.n_qubits)
        return (prefactor * torch.einsum("if,if->i", v0, v0)).detach().cpu().numpy()

    def is_stationary(self) -> bool:
        return False

    def get_params(self, deep: bool = True) -> dict:
        return {"kernel": self.kernel, "scale": self.scale}

    def __sklearn_clone__(self) -> "SklearnNIFCovarianceKernel":
        """Return the wrapper itself rather than a reconstruction of it.

        :return: This wrapper.
        :rtype: SklearnNIFCovarianceKernel
        """
        return self

    @property
    def hyperparameters(self):
        return []

    @property
    def theta(self) -> np.ndarray:
        return np.array([])

    @theta.setter
    def theta(self, value) -> None:
        pass  # no tunable sklearn hyperparameters

    @property
    def bounds(self) -> np.ndarray:
        return np.empty((0, 2))

    def _gram(self, X: np.ndarray, Y: Optional[np.ndarray]) -> torch.Tensor:
        v0 = self.kernel.feature_map(X)
        v1 = v0 if Y is None else self.kernel.feature_map(Y)
        prefactor = self.scale * self.kernel.observable_norm / (2**self.kernel.n_qubits)
        return prefactor * torch.einsum("if,jf->ij", v0, v1)

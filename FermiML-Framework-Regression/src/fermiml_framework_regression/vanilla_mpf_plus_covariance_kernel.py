from .multi_sector_plus_kernel import MultiSectorPlusCovarianceKernel
from .vanilla_nif_covariance_kernel import VanillaNIFCovarianceKernel


class VanillaMpfPlusCovarianceKernel(MultiSectorPlusCovarianceKernel, VanillaNIFCovarianceKernel):
    r"""Multi-sector covariance kernel with the trainable linear ansatz, on :math:`\ket{+}^{\otimes n}`.

    :class:`~fermiml_framework_regression.vanilla_nif_covariance_kernel.VanillaNIFCovarianceKernel`
    with the :class:`~fermiml_framework_regression.multi_sector_plus_kernel.MultiSectorPlusCovarianceKernel`
    feature map: the same ``LazyLinear`` encoder driving one global SPTM, but starting from
    :math:`\ket{+}^{\otimes n}` and reading several Majorana orders instead of one.

    Method resolution order:
    ``VanillaMpfPlusCovarianceKernel -> MultiSectorPlusCovarianceKernel -> VanillaNIFCovarianceKernel
    -> NIFCovarianceKernel -> LinearNIFKernel -> NIFKernel -> Kernel``.

    :param sectors: Majorana orders to span.
    """

    @VanillaNIFCovarianceKernel.n_qubits.setter  # type: ignore[attr-defined]
    def n_qubits(self, value: int) -> None:
        VanillaNIFCovarianceKernel.n_qubits.fset(self, value)  # type: ignore[attr-defined]
        self.__dict__.pop("index_sets_per_order", None)

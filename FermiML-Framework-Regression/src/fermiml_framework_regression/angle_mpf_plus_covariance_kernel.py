from .angle_nif_covariance_kernel import AngleNIFCovarianceKernel
from .multi_sector_plus_kernel import MultiSectorPlusCovarianceKernel


class AngleMpfPlusCovarianceKernel(MultiSectorPlusCovarianceKernel, AngleNIFCovarianceKernel):
    r"""Multi-sector covariance kernel with the fixed angle ansatz, on :math:`\ket{+}^{\otimes n}`.

    :class:`~fermiml_framework_regression.angle_nif_covariance_kernel.AngleNIFCovarianceKernel`
    with the :class:`~fermiml_framework_regression.multi_sector_plus_kernel.MultiSectorPlusCovarianceKernel`
    feature map: the same parameter-free ``SptmAngleEmbedding``, but starting from
    :math:`\ket{+}^{\otimes n}` and reading several Majorana orders instead of one. Having no
    trainable parameters, the kernel is still frozen at construction.

    Method resolution order:
    ``AngleMpfPlusCovarianceKernel -> MultiSectorPlusCovarianceKernel -> AngleNIFCovarianceKernel
    -> NIFCovarianceKernel -> NIFKernel -> Kernel``.

    :param sectors: Majorana orders to span.
    :param rotations: Comma-separated rotation letters for the angle embedding.
    """

    @AngleNIFCovarianceKernel.n_qubits.setter  # type: ignore[attr-defined]
    def n_qubits(self, value: int) -> None:
        # The parent setter invalidates the single-order index sets; the per-order arrays are
        # keyed on n_qubits just as much.
        AngleNIFCovarianceKernel.n_qubits.fset(self, value)  # type: ignore[attr-defined]
        self.__dict__.pop("index_sets_per_order", None)

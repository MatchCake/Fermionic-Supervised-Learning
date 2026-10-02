from matchcake.operations import SptmAngleEmbedding

from .nif_covariance_kernel import NIFCovarianceKernel


class AngleNIFCovarianceKernel(NIFCovarianceKernel):
    r"""Matchgate covariance kernel with a fixed angle-embedding ansatz.

    Implements the standard ``quantum kernel`` instantiation of the matchgate
    covariance kernel of arXiv:2605.00099: the input :math:`x` is embedded
    directly as rotation angles via
    :class:`~matchcake.operations.SptmAngleEmbedding`, with no trainable
    parameters. The feature map is therefore fixed at construction and the
    kernel can be ``.freeze()``-d immediately.

    .. note::
        Matchgate angle embeddings pair features to adjacent wires, so this
        ansatz expects an even ``n_qubits`` when the number of features equals
        ``n_qubits``.

    :param rotations: Comma-separated rotation letters applied per wire pair,
        e.g. ``"Y,X"`` for an :math:`R_y` then :math:`R_x` layer.
    """

    DEFAULT_ROTATIONS = "Y,X"

    def __init__(self, *, rotations: str = DEFAULT_ROTATIONS, **kwargs):
        super().__init__(**kwargs)
        self.rotations = rotations

    def ansatz(self, x):
        yield from SptmAngleEmbedding(x, wires=self.wires, rotations=self.rotations).decomposition()

from matchcake.ml import LinearNIFKernel

from .nif_covariance_kernel import NIFCovarianceKernel


class VanillaNIFCovarianceKernel(NIFCovarianceKernel, LinearNIFKernel):
    r"""Matchgate covariance kernel with the linear (Vanilla) ansatz.

    Combines :class:`NIFCovarianceKernel` with :class:`LinearNIFKernel`: the
    input is pushed through a single :class:`torch.nn.LazyLinear` layer whose
    output fills the upper triangle of an antisymmetric :math:`2n \times 2n`
    Hamiltonian generator, which is exponentiated into one global SPTM. The
    resulting covariance feature map is used in the matchgate GP kernel of
    arXiv:2605.00099.

    Method resolution order:
    ``VanillaNIFCovarianceKernel -> NIFCovarianceKernel -> LinearNIFKernel
    -> NIFKernel -> Kernel``.

    .. note::
        The ansatz uses :class:`torch.nn.LazyLinear`, whose parameters
        materialise on the first forward pass. Call :meth:`fit` before
        attempting to freeze the kernel.

    :param sector_m: Even Majorana sector :math:`m`.
    :param observable_norm: Multiplicative scale :math:`c` on the kernel.
    :param bias: Whether the :class:`~torch.nn.LazyLinear` encoder has a bias.
    :param encoder_activation: Activation applied after the linear layer.
    """

    def __init__(
        self,
        *,
        sector_m: int = NIFCovarianceKernel.DEFAULT_SECTOR_M,
        observable_norm: float = NIFCovarianceKernel.DEFAULT_OBSERVABLE_NORM,
        gram_batch_size: int = LinearNIFKernel.DEFAULT_GRAM_BATCH_SIZE,
        random_state: int = LinearNIFKernel.DEFAULT_RANDOM_STATE,
        alignment: bool = LinearNIFKernel.DEFAULT_ALIGNMENT,
        alignment_iterations: int = LinearNIFKernel.DEFAULT_ALIGNMENT_ITERATIONS,
        alignment_learning_rate: float = LinearNIFKernel.DEFAULT_ALIGNMENT_LEARNING_RATE,
        alignment_early_stopping_patience: int = LinearNIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_PATIENCE,
        alignment_early_stopping_threshold: float = LinearNIFKernel.DEFAULT_ALIGNMENT_EARLY_STOPPING_THRESHOLD,
        n_qubits: int = LinearNIFKernel.DEFAULT_N_QUBITS,
        bias: bool = LinearNIFKernel.DEFAULT_BIAS,
        encoder_activation: str = LinearNIFKernel.DEFAULT_ENCODER_ACTIVATION,
    ):
        # The two parents take disjoint kwargs, so cooperative super() would
        # force awkward argument juggling. Build LinearNIFKernel state directly
        # (device + encoder), then layer on the covariance-kernel state.
        LinearNIFKernel.__init__(
            self,
            gram_batch_size=gram_batch_size,
            random_state=random_state,
            alignment=alignment,
            alignment_iterations=alignment_iterations,
            alignment_learning_rate=alignment_learning_rate,
            alignment_early_stopping_patience=alignment_early_stopping_patience,
            alignment_early_stopping_threshold=alignment_early_stopping_threshold,
            n_qubits=n_qubits,
            bias=bias,
            encoder_activation=encoder_activation,
        )
        if sector_m % 2 != 0 or sector_m < 2:
            raise ValueError(f"sector_m must be a positive even integer; got {sector_m}.")
        if sector_m > 2 * self.n_qubits - 2:
            raise ValueError(
                f"sector_m={sector_m} exceeds 2n-2={2 * self.n_qubits - 2}; no non-trivial m-subsets exist."
            )
        self._sector_m = sector_m
        self.observable_norm = observable_norm

    @property
    def n_qubits(self) -> int:
        return len(self._q_device.wires)

    @n_qubits.setter
    def n_qubits(self, value: int) -> None:
        # Rebuild the device and the LazyLinear encoder (LinearNIFKernel
        # behaviour), then invalidate the covariance index-set cache.
        LinearNIFKernel.n_qubits.fset(self, value)  # type: ignore[union-attr]
        self.__dict__.pop("index_sets", None)

from typing import Union

HILBERT_DIMENSION_NORM = "2^n"


def kernel_prefactor(observable_norm: Union[float, str], n_qubits: int) -> float:
    r"""Prefactor :math:`c / 2^n` of the order-m Pfaffian covariance kernel.

    :param observable_norm: The scale :math:`c`, or :data:`HILBERT_DIMENSION_NORM` for :math:`c = 2^n`.
    :param n_qubits: Number of qubits :math:`n`.
    :return: The prefactor multiplying :math:`v_m(x)^\top v_m(x')`.
    :rtype: float
    """
    if observable_norm == HILBERT_DIMENSION_NORM:
        return 1.0
    if isinstance(observable_norm, bool) or not isinstance(observable_norm, (int, float)):
        raise ValueError(f"observable_norm must be a number or {HILBERT_DIMENSION_NORM!r}; got {observable_norm!r}.")
    return float(observable_norm) * 2.0**-n_qubits

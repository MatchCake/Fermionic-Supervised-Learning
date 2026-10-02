from functools import cached_property
from typing import Tuple

import numpy as np
import pennylane as qml
from matchcake import NonInteractingFermionicDevice
from matchcake.operations import SptmAngleEmbedding

from .dataset import Dataset


class MagnetisationDataset(Dataset):
    r"""Total magnetisation of a matchgate-embedded state.

    The inputs are angles :math:`x \sim \mathcal{U}(0, \pi)^{n}` and the target is

    .. math::

        y(x) = \Bigl\langle \sum_{k=1}^{n} Z_k \Bigr\rangle_{\rho(x)},

    where :math:`\rho(x)` is prepared by a one-layer
    :class:`~matchcake.operations.SptmAngleEmbedding` of :math:`x`. Each
    :math:`\langle Z_k \rangle` is a two-point Majorana observable, so the target lies
    **exactly** in the hypothesis class of the order-:math:`m=2` covariance kernel when
    the model uses the same embedding and the same qubit count.

    That is the point of this dataset. Friedman and California Housing carry no
    free-fermion structure and measure whether the matchgate kernel is competitive on
    generic data; this one measures what it does on data it is provably suited to, and is
    the reference against which any advantage claim on the other two has to be read. The
    advantage is conditional by construction, not general.

    Targets are computed on :class:`~matchcake.NonInteractingFermionicDevice`, which
    reproduces the exact state-vector expectation to ~1e-6 in float32.

    Two consequences of the construction worth knowing before gridding it:

    * The number of features equals ``n_qubits``, so a model configured with fewer qubits
      than the dataset raises from :class:`~matchcake.operations.SptmAngleEmbedding`
      (features must fit on the wires). A model with more qubits is fine: the embedding
      uses only the first ``n_qubits`` wires, but the extra wires dilute the
      :math:`2^{-n}` kernel prefactor.
    * With ``noise=0`` the target is deterministic given ``x``, so the Bayes error is
      zero and a test :math:`R^2` below 1 is entirely model limitation. Raise ``noise``
      to put a ceiling on it.

    :param n_qubits: Number of qubits, which is also the input dimension.
    :param n_samples: Number of training-pool samples.
    :param n_test_samples: Number of held-out test samples. Drawn with a different seed
        (``random_state + 1000``) so the test set is independent of the training pool.
    :param noise: Standard deviation of the Gaussian noise added to the training targets.
        The test targets are always noiseless so they reflect the clean ground truth.
    :param rotations: Comma-separated rotation letters of the embedding, e.g. ``"Y,X"``.
        A model can only be exactly in the RKHS if it uses the same value.
    :param random_state: Base random seed for reproducibility.
    """

    DATASET_NAME = "magnetisation"
    DEFAULT_N_QUBITS = 6
    DEFAULT_N_SAMPLES = 500
    DEFAULT_N_TEST_SAMPLES = 100
    DEFAULT_NOISE = 0.0
    DEFAULT_ROTATIONS = "Y,X"
    DEFAULT_RANDOM_STATE = 0

    def __init__(
        self,
        *,
        n_qubits: int = DEFAULT_N_QUBITS,
        n_samples: int = DEFAULT_N_SAMPLES,
        n_test_samples: int = DEFAULT_N_TEST_SAMPLES,
        noise: float = DEFAULT_NOISE,
        rotations: str = DEFAULT_ROTATIONS,
        random_state: int = DEFAULT_RANDOM_STATE,
        cv_split_id: int = Dataset.DEFAULT_CV_SPLIT_ID,
        cv_splits: int = Dataset.DEFAULT_CV_SPLITS,
        cv_random_state: int = Dataset.DEFAULT_CV_RANDOM_STATE,
        val_size: float = Dataset.DEFAULT_VAL_SIZE,
    ):
        super().__init__(
            cv_split_id=cv_split_id,
            cv_splits=cv_splits,
            cv_random_state=cv_random_state,
            val_size=val_size,
        )
        if n_qubits < 2 or n_qubits % 2 != 0:
            raise ValueError(f"n_qubits must be an even integer >= 2; got {n_qubits}.")
        if noise < 0:
            raise ValueError(f"noise must be non-negative; got {noise}.")
        self.n_qubits = n_qubits
        self.n_samples = n_samples
        self.n_test_samples = n_test_samples
        self.noise = noise
        self.rotations = rotations
        self.random_state = random_state

    def fetch_x_train(self) -> np.ndarray:
        """:return: Training-pool inputs of shape ``(n_samples, n_qubits)``. :rtype: np.ndarray"""
        return self._train_data[0]

    def fetch_y_train(self) -> np.ndarray:
        """:return: Training-pool targets of shape ``(n_samples,)``. :rtype: np.ndarray"""
        return self._train_data[1]

    def fetch_x_test(self) -> np.ndarray:
        """:return: Test inputs of shape ``(n_test_samples, n_qubits)``. :rtype: np.ndarray"""
        return self._test_data[0]

    def fetch_y_test(self) -> np.ndarray:
        """:return: Noiseless test targets of shape ``(n_test_samples,)``. :rtype: np.ndarray"""
        return self._test_data[1]

    def _generate(self, n_samples: int, random_state: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
        """Draw inputs and evaluate the magnetisation of the embedded state.

        The whole batch goes through the device in one call: the covariance matrices are
        ``(n_samples, 2n, 2n)``, which is small next to the kernel feature maps built
        from them.

        :param n_samples: Number of samples to draw.
        :param random_state: Seed for both the inputs and the target noise.
        :param noise: Standard deviation of the Gaussian noise added to the targets.
        :return: The inputs and their targets.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        rng = np.random.default_rng(random_state)
        x = rng.uniform(0.0, np.pi, size=(n_samples, self.n_qubits)).astype(np.float32)
        device = NonInteractingFermionicDevice(self.n_qubits)
        observable = qml.sum(*[qml.Z(k) for k in range(self.n_qubits)])
        values = device.execute_generator(
            SptmAngleEmbedding(x, wires=range(self.n_qubits), rotations=self.rotations).decomposition(),
            observable=observable,
            output_type="expval",
            reset=True,
        )
        y = np.real(np.asarray(values)).ravel().astype(np.float32)
        if noise > 0:
            y = (y + rng.normal(0.0, noise, size=y.shape)).astype(np.float32)
        return x, y

    @cached_property
    def _train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """:return: The training pool, noise included. :rtype: Tuple[np.ndarray, np.ndarray]"""
        return self._generate(self.n_samples, self.random_state, self.noise)

    @cached_property
    def _test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """:return: The held-out test set, always noiseless. :rtype: Tuple[np.ndarray, np.ndarray]"""
        return self._generate(self.n_test_samples, self.random_state + 1000, 0.0)

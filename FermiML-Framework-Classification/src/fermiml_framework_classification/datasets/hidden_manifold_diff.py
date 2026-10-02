from typing import Optional, Union

import numpy as np

from .dataset import Dataset, _load_hdf5_split


class HiddenManifoldDiff(Dataset):
    DATASET_NAME = "hidden-manifold-diff"
    _HDF5_NAME = "hidden-manifold"
    FEATURE_DIMENSION = 10

    @classmethod
    def feature_dimension(cls, extra_options: Optional[dict] = None) -> int:
        """Always 10, whatever ``n_manifolds`` is.

        Unlike its sibling :class:`~.hidden_manifold.HiddenManifold`, this dataset sweeps the number of latent
        manifolds and not the ambient dimension: every ``diff_train``/``diff_test`` group in the HDF5 file
        carries 10 features.

        :param extra_options: the dataset's ``extra_options``; unused, kept for interface symmetry.
        :return: the number of input features.
        :rtype: int
        """
        return cls.FEATURE_DIMENSION

    def __init__(
        self,
        *,
        n_manifolds: Union[int, str] = 20,
        **kwargs,
    ):
        assert 2 <= int(n_manifolds) <= 20, "Number of n_manifolds must be between 2 and 20"
        super().__init__(**kwargs)
        self._n_manifolds = str(n_manifolds)

    def fetch_x_train(self) -> np.ndarray:
        inputs, _ = _load_hdf5_split(self._HDF5_NAME, "diff_train", self._n_manifolds)
        return inputs

    def fetch_y_train(self) -> np.ndarray:
        _, labels = _load_hdf5_split(self._HDF5_NAME, "diff_train", self._n_manifolds)
        return labels

    def fetch_x_test(self) -> np.ndarray:
        inputs, _ = _load_hdf5_split(self._HDF5_NAME, "diff_test", self._n_manifolds)
        return inputs

    def fetch_y_test(self) -> np.ndarray:
        _, labels = _load_hdf5_split(self._HDF5_NAME, "diff_test", self._n_manifolds)
        return labels

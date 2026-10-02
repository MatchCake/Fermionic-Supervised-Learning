import warnings

import importlib_metadata

__author__ = "Jeremie Gince"
__email__ = "Jeremie.Gicne@usherbrooke.ca"
__copyright__ = "Copyright 2026, Jeremie Gince"
__license__ = "Apache 2.0"
__url__ = "https://github.com/MatchCake/FermiML-Framework-Regression"
__package__ = "fermiml_framework_regression"
__version__ = importlib_metadata.version(__package__)

warnings.filterwarnings("ignore", category=Warning, module="docutils")
warnings.filterwarnings("ignore", category=Warning, module="sphinx")

from .angle_nif_covariance_kernel import AngleNIFCovarianceKernel
from .nif_covariance_kernel import NIFCovarianceKernel
from .sklearn_nif_covariance_kernel import SklearnNIFCovarianceKernel
from .vanilla_nif_covariance_kernel import VanillaNIFCovarianceKernel

__all__ = [
    "NIFCovarianceKernel",
    "VanillaNIFCovarianceKernel",
    "AngleNIFCovarianceKernel",
    "SklearnNIFCovarianceKernel",
]

from .angle_mpf_plus_gp_regressor import AngleMpfPlusGPRegressor
from .angle_nif_gp_regressor import AngleNIFGPRegressor
from .classical_gp_regressor import ClassicalGPRegressor
from .classical_svr import ClassicalSVR
from .nif_gp_regressor import NIFGPRegressor
from .nif_kernel_regressor import NIFKernelRegressor
from .vanilla_mpf_plus_gp_regressor import VanillaMpfPlusGPRegressor
from .vanilla_nif_gp_regressor import VanillaNIFGPRegressor
from .vanilla_nif_kernel_regressor import VanillaNIFKernelRegressor

__all__ = [
    "NIFKernelRegressor",
    "VanillaNIFKernelRegressor",
    "NIFGPRegressor",
    "AngleNIFGPRegressor",
    "VanillaNIFGPRegressor",
    "AngleMpfPlusGPRegressor",
    "VanillaMpfPlusGPRegressor",
    "ClassicalGPRegressor",
    "ClassicalSVR",
]

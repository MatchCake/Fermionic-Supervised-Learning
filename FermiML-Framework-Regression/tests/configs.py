from pathlib import Path

import numpy as np

CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "configs" / "config.yaml"

ATOL_SCALAR_COMPARISON = 1e-6
RTOL_SCALAR_COMPARISON = 1e-6

ATOL_MATRIX_COMPARISON = 1e-5
RTOL_MATRIX_COMPARISON = 1e-5

MIN_RKHS_TEST_R2 = 0.99

DEFAULT_SEED = 0
DEFAULT_RNG = np.random.default_rng(DEFAULT_SEED)

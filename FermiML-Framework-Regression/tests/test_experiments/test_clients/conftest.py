import pytest
import yaml

# The smallest grid that still exercises the real expansion: two models over two CV splits,
# on the cheapest dataset. Kept as YAML rather than built in Python because the CLIs read a
# file, and the kebab-case keys the serializer expects are part of what is under test.
TEST_CONFIG = {
    "experiment-name": "cli_test_experiment",
    "datasets": {
        "cv-splits": 2,
        "splits-seed": 0,
        "val-size": 0.25,
        "datasets": [
            {
                "name": "magnetisation",
                "extra-options": {},
                "hyperparameters": {"n_qubits": [4], "n_samples": [32], "n_test_samples": [12]},
            }
        ],
    },
    "models": [
        {
            "name": "AngleNIFGP",
            "extra-options": {},
            "hyperparameters": {"n_qubits": [-1], "sector_m": [2]},
            "slurm-resources": {"mem_mb": 8192, "runtime_min": 60},
        },
        {
            "name": "ClassicalGP",
            "extra-options": {},
            "hyperparameters": {"kernel_length_scale": [1.0]},
            "slurm-resources": {"mem_mb": 4096, "runtime_min": 30},
        },
    ],
}

EXPECTED_JOBS = 4  # 2 CV splits x 2 model configurations


@pytest.fixture
def config_file(tmp_path):
    """Write the test grid to a YAML file and return its path.

    :param tmp_path: pytest temporary directory.
    :return: Path to the written config.
    """
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(TEST_CONFIG))
    return path

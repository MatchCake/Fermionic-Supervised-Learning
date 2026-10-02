import os
from pathlib import Path

import click
from pythonbasictools import OutputFolder
from pythonbasictools.experiment_utils.output_folder import ExperimentState

from ..configs.experiment_config import ExperimentConfig


@click.command()
@click.option("--exp_config_file", type=click.Path(exists=True), envvar="EXP_CONFIG_FILE", default=None)
@click.option("--root_dir", type=click.Path(), envvar="DATA_DIR", default="data")
@click.option("--out_file", type=click.Path(), envvar="N_OUT_FILE", default="n.txt")
def compute_n_cli(
    exp_config_file: Path,
    root_dir: Path,
    out_file: Path,
) -> int:
    """Count how many experiments are still unfinished.

    :param exp_config_file: Path to a YAML configuration file.
    :param root_dir: Root folder for experiment outputs.
    :param out_file: File to write the count into.
    """
    exp_config_file = Path(exp_config_file)
    exp_config: ExperimentConfig = ExperimentConfig.from_yaml_file(exp_config_file)  # type: ignore
    click.echo(f"Running compute_n_cli from config at {exp_config_file.resolve()} with:\n{exp_config}")
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", -1)
    os.environ["OMP_NUM_THREADS"] = str(slurm_cpus)
    os.environ["MKL_NUM_THREADS"] = str(slurm_cpus)
    root_output_folder = Path(root_dir) / exp_config.experiment_name / "results"

    configs = sorted(exp_config.to_single_experiment_configs(), key=lambda x: x.hash)
    unfinished = [
        cfg for cfg in configs if OutputFolder(Path(root_output_folder) / cfg.hash).state != ExperimentState.FINISHED
    ]
    with open(out_file, "w") as f:
        f.write(str(len(unfinished)))
    click.echo(str(len(unfinished)))
    return 0

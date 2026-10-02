import os
from pathlib import Path

import click
from pythonbasictools import OutputFolder
from pythonbasictools.experiment_utils.output_folder import ExperimentState

from ..configs.experiment_config import ExperimentConfig
from ..single_experiment import SingleExperiment


@click.command()
@click.option("--exp_config_file", type=click.Path(exists=True), envvar="EXP_CONFIG_FILE", default=None)
@click.option("--root_dir", type=click.Path(), envvar="DATA_DIR", default="data")
@click.option("--out_file", type=click.Path(), envvar="N_OUT_FILE", default="n.txt")
def compute_n_cli(
    exp_config_file: Path,
    root_dir: Path,
    out_file: Path,
):
    """
    Compute how many experiments are left to run from an experiment config.

    :param exp_config_file: Path to a YAML configuration file.
    :param root_dir: Path to the root folder of the output folder. This is used to determine where to save the results
        of the experiment.
    :param out_file: Path to the output file where the number of experiments is written.
    """
    exp_config_file = Path(exp_config_file)
    exp_config: ExperimentConfig = ExperimentConfig.from_yaml_file(exp_config_file)  # type: ignore
    click.echo(f"Running compute_ith_cli from config at {exp_config_file.resolve()} with:\n{exp_config}")
    SLURM_CPUS_PER_TASK = os.environ.get("SLURM_CPUS_PER_TASK", -1)
    os.environ["OMP_NUM_THREADS"] = str(SLURM_CPUS_PER_TASK)
    os.environ["MKL_NUM_THREADS"] = str(SLURM_CPUS_PER_TASK)
    click.echo(f"Setting OMP_NUM_THREADS={SLURM_CPUS_PER_TASK} from SLURM_CPUS_PER_TASK")
    click.echo(f"Setting MKL_NUM_THREADS={SLURM_CPUS_PER_TASK} from SLURM_CPUS_PER_TASK")
    root_output_folder = Path(root_dir) / exp_config.experiment_name / "results"

    configs = exp_config.to_single_experiment_configs()
    configs = list(sorted(configs, key=lambda x: x.hash))
    configs = [
        cfg for cfg in configs if OutputFolder(Path(root_output_folder) / cfg.hash).state != ExperimentState.FINISHED
    ]
    with open(out_file, "w") as f:
        f.write(str(len(configs)))
    click.echo(str(len(configs)))
    return 0

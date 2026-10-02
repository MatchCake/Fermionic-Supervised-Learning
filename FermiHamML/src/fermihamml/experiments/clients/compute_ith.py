import os
from pathlib import Path

import click

from ..single_experiment import SingleExperiment
from ..configs.experiment_config import ExperimentConfig


@click.command()
@click.argument("config_id", type=int, nargs=1)
@click.argument("--exp_config_file", type=click.Path(exists=True), envvar="EXP_CONFIG_FILE", default=None)
@click.option("--root_dir", type=click.Path(), envvar="DATA_DIR", default=None)
def compute_ith_cli(
    config_id: int,
    exp_config_file: Path,
    root_dir: Path,
):
    """
    Compute the ith experiment from an experiment config.

    :param config_id: The index of the experiment config to compute.
        This should be an integer between 0 and the total number of experiment configs - 1.
    :param exp_config_file: Path to a YAML configuration file.
    :param root_dir: Path to the root folder of the output folder. This is used to determine where to save the results
        of the experiment.
    """
    exp_config_file = Path(exp_config_file)
    exp_config: ExperimentConfig = ExperimentConfig.from_yaml_file(exp_config_file)  # type: ignore
    click.echo(f"Running compute_ith_cli from config at {exp_config_file.resolve()} with:\n{exp_config}")
    SLURM_CPUS_PER_TASK = os.environ.get("SLURM_CPUS_PER_TASK", -1)
    os.environ["OMP_NUM_THREADS"] = str(SLURM_CPUS_PER_TASK)
    os.environ["MKL_NUM_THREADS"] = str(SLURM_CPUS_PER_TASK)
    click.echo(f"Setting OMP_NUM_THREADS={SLURM_CPUS_PER_TASK} from SLURM_CPUS_PER_TASK")
    click.echo(f"Setting MKL_NUM_THREADS={SLURM_CPUS_PER_TASK} from SLURM_CPUS_PER_TASK")
    configs = exp_config.to_single_experiment_configs()
    configs = list(sorted(configs, key=lambda x: x.hash))
    config = configs[config_id]

    click.echo(f"Running single experiment with the following config:\n{config}")
    click.echo(f"Computing ith experiment config with hash {config.hash}...")
    root_output_folder = Path(root_dir) / exp_config.experiment_name / "results"
    experiment = SingleExperiment(
        config,
        root_output_folder=root_output_folder,
        output_folder=config.hash,
        raise_errors=False,
    )  # type: ignore
    experiment.run()
    click.echo(f"Finished computing ith experiment config with hash {config.hash}.")
    click.echo(f"Finished compute_ith_cli.")
    return 0

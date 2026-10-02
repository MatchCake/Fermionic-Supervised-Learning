import os
from pathlib import Path

import click

from ..configs.single_experiment_config import SingleExperimentConfig
from ..single_experiment import SingleExperiment


@click.command()
@click.argument("config_file", type=click.Path(exists=True), nargs=1)
@click.option("--root_output_folder", type=click.Path(), envvar="DATA_DIR", default=None)
@click.option("--output_folder", type=click.Path(), default=None)
@click.option("--raise_errors", is_flag=True, default=False, help="Whether to raise errors or just log them.")
def single_experiment_cli(
    config_file: Path,
    root_output_folder: Path,
    output_folder: Path,
    raise_errors: bool,
):
    """
    Run a single experiment.

    :param config_file: Path to a YAML configuration file.
    :param root_output_folder: Path to the root folder of the output folder.
    :param output_folder: Path to the output folder.
    :param raise_errors: Whether to raise errors or just log them.
    """
    config_file = Path(config_file)
    config: SingleExperimentConfig = SingleExperimentConfig.from_yaml_file(config_file)  # type: ignore
    click.echo(f"Running single_experiment_cli from config at {config_file.resolve()} with:\n{config}")
    SLURM_CPUS_PER_TASK = os.environ.get("SLURM_CPUS_PER_TASK", -1)
    os.environ["OMP_NUM_THREADS"] = str(SLURM_CPUS_PER_TASK)
    os.environ["MKL_NUM_THREADS"] = str(SLURM_CPUS_PER_TASK)
    click.echo(f"Setting OMP_NUM_THREADS={SLURM_CPUS_PER_TASK} from SLURM_CPUS_PER_TASK")
    click.echo(f"Setting MKL_NUM_THREADS={SLURM_CPUS_PER_TASK} from SLURM_CPUS_PER_TASK")
    experiment = SingleExperiment(
        config,
        root_output_folder=root_output_folder,
        output_folder=output_folder,
        raise_errors=raise_errors,
    )  # type: ignore
    experiment.run()
    click.echo(f"Finished single_experiment_cli.")
    return 0

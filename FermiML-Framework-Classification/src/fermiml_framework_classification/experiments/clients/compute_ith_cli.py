import os
from pathlib import Path

import click
from pythonbasictools import OutputFolder
from pythonbasictools.experiment_utils.output_folder import ExperimentState

from ..configs.experiment_config import ExperimentConfig
from ..single_experiment import SingleExperiment


@click.command()
@click.argument("config_id", type=int, nargs=1, required=False, default=None)
@click.option("--config_hash", type=str, default=None, help="Hash of the experiment config to compute.")
@click.option("--exp_config_file", type=click.Path(exists=True), envvar="EXP_CONFIG_FILE", default=None)
@click.option("--root_dir", type=click.Path(), envvar="DATA_DIR", default="data")
@click.option("--raise_errors", is_flag=True, default=False, help="Whether to raise errors or just log them.")
def compute_ith_cli(
    config_id: int,
    config_hash: str,
    exp_config_file: Path,
    root_dir: Path,
    raise_errors: bool,
):
    """
    Compute a single experiment from an experiment config, identified either by
    its integer index (CONFIG_ID) or by --config_hash.
    """
    if config_id is None and config_hash is None:
        raise click.UsageError("Provide either CONFIG_ID or --config_hash.")
    exp_config_file = Path(exp_config_file)
    exp_config: ExperimentConfig = ExperimentConfig.from_yaml_file(exp_config_file)  # type: ignore
    click.echo(f"Running compute_ith_cli from config at {exp_config_file.resolve()} with:\n{exp_config}")
    SLURM_CPUS_PER_TASK = os.environ.get("SLURM_CPUS_PER_TASK", -1)
    os.environ["OMP_NUM_THREADS"] = str(SLURM_CPUS_PER_TASK)
    os.environ["MKL_NUM_THREADS"] = str(SLURM_CPUS_PER_TASK)
    click.echo(f"Setting OMP_NUM_THREADS={SLURM_CPUS_PER_TASK} from SLURM_CPUS_PER_TASK")
    click.echo(f"Setting MKL_NUM_THREADS={SLURM_CPUS_PER_TASK} from SLURM_CPUS_PER_TASK")
    root_output_folder = Path(root_dir) / exp_config.experiment_name / "results"

    configs = list(sorted(exp_config.to_single_experiment_configs(), key=lambda x: x.hash))
    if config_hash is not None:
        matched = [c for c in configs if c.hash == config_hash]
        if not matched:
            raise ValueError(f"No config found with hash {config_hash!r}")
        config = matched[0]
    else:
        if config_id >= len(configs):
            raise ValueError(f"config_id {config_id} is out of range for the number of configs {len(configs)}")
        config = configs[config_id]

    click.echo(f"Running single experiment with the following config:\n{config}")
    click.echo(f"Computing ith experiment config with hash {config.hash}...")

    experiment = SingleExperiment(
        config,
        root_output_folder=root_output_folder,
        output_folder=config.hash,
        raise_errors=raise_errors,
    )  # type: ignore
    experiment.run()
    click.echo(f"Finished computing ith experiment config with hash {config.hash}.")
    click.echo(f"Finished compute_ith_cli.")
    return 0

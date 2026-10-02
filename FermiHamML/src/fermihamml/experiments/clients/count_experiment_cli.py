import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import click
from tqdm import tqdm

from fermihamml.experiments.configs.experiment_config import (
    ExperimentConfig,
)
from fermihamml.utils.output_folder import ExperimentState, OutputFolder


@click.command()
@click.argument("config_file", type=click.Path(exists=True), nargs=1)
@click.option("--root_dir", type=click.Path(), default="data")
def count_experiment_cli(
    config_file: Path,
    root_dir: Path,
):
    """
    Run count of the experiment.

    :param config_file: Path to a YAML configuration file.
    :param root_dir: Path to the root folder of the output folder.
    """
    config_file = Path(config_file)
    root_dir = Path(root_dir)
    config = ExperimentConfig.from_yaml_file(config_file)
    click.echo(f"Running count_experiment_cli from config at {config_file.resolve()} with:\n{config}")
    SLURM_CPUS_PER_TASK = os.environ.get("SLURM_CPUS_PER_TASK", -1)
    os.environ["OMP_NUM_THREADS"] = str(SLURM_CPUS_PER_TASK)
    os.environ["MKL_NUM_THREADS"] = str(SLURM_CPUS_PER_TASK)
    click.echo(f"Setting OMP_NUM_THREADS={SLURM_CPUS_PER_TASK} from SLURM_CPUS_PER_TASK")
    click.echo(f"Setting MKL_NUM_THREADS={SLURM_CPUS_PER_TASK} from SLURM_CPUS_PER_TASK")
    jobs_stats_filepath = root_dir / config.experiment_name / "states_stats.json"  # type: ignore
    output_folders = OutputFolder.gather_output_folders(root_dir / config.experiment_name)  # type: ignore
    counter = Counter(of.state.name for of in output_folders)

    failed_of = [of for of in output_folders if of.state == ExperimentState.FAILED]
    failed_msgs = defaultdict(list)
    for of in tqdm(failed_of, desc="Collecting failed messages"):
        failed_msgs[of.state_file.read()].append(str(of.path))

    jobs_stats_filepath.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "num_jobs": len(output_folders),
        "states": counter,
        "failed_msgs": dict(failed_msgs),
    }
    with open(jobs_stats_filepath, "w") as f:
        json.dump(counts, f, indent=4)
    click.echo("Counts of jobs states:")
    click.echo(json.dumps(counts, indent=4, default=str))
    click.echo(f"Finished count_experiment_cli.")
    return 0

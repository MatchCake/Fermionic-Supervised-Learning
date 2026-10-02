import json
import os
from collections import Counter
from pathlib import Path
from typing import List

import click

from ...utils.output_folder import ExperimentState, OutputFolder
from ..configs.experiment_config import ExperimentConfig


def get_state_counter(job_paths: List[Path]) -> Counter:
    return Counter(OutputFolder(job_path).state.name for job_path in job_paths)


@click.command()
@click.argument("config_file", type=click.Path(exists=True), nargs=1)
@click.option("--root_dir", type=click.Path(), envvar="DATA_DIR", default=None)
@click.option("--cmd", type=str, default="uv")
@click.option("--script", type=str, default="single_exp")
@click.option("--dry_run", is_flag=True, default=False)
@click.option("--out_file", type=click.Path(), default=None)
def launch_jobs_cli(
    config_file: Path,
    root_dir: Path,
    cmd: str,
    script: str,
    dry_run: bool,
    out_file: Path,
):
    """
    Prepare experiment configs yaml files.

    :param config_file: Path to a YAML configuration file.
    :param root_dir: Path to the root folder of the configs yaml files.
    :param cmd: Command to use for submitting jobs. Choices are "uv" for using uv and "sbatch" for using sbatch.
    :param script: Script to run for each job. This should be the name of a script in the "scripts" folder.
    :param dry_run: If set, only prepare the yaml files but do not submit the jobs. This is useful for debugging.
    :param out_file: If set, write the commands to this file instead of executing them. This is useful for debugging
        and for running the commands manually later.
    """
    config_file = Path(config_file)
    config: ExperimentConfig = ExperimentConfig.from_yaml_file(config_file)  # type: ignore
    click.echo(f"Running launch_jobs_cli from config at {config_file.resolve()} with:\n{config}")
    SLURM_CPUS_PER_TASK = os.environ.get("SLURM_CPUS_PER_TASK", -1)
    os.environ["OMP_NUM_THREADS"] = str(SLURM_CPUS_PER_TASK)
    os.environ["MKL_NUM_THREADS"] = str(SLURM_CPUS_PER_TASK)
    click.echo(f"Setting OMP_NUM_THREADS={SLURM_CPUS_PER_TASK} from SLURM_CPUS_PER_TASK")
    click.echo(f"Setting MKL_NUM_THREADS={SLURM_CPUS_PER_TASK} from SLURM_CPUS_PER_TASK")
    click.echo("Preparing experiment configs yaml files...")
    config_paths = config.to_single_experiment_yamls(root_dir=root_dir)
    configs = config.to_single_experiment_configs()  # type: ignore
    click.echo(f"Finished preparing experiment configs yaml files.")
    job_paths = [p.parents[1] / "results" / f"{p.stem}" for p in config_paths]
    jobs_stats_filepath = Path(root_dir) / config.experiment_name / "jobs_stats.json"
    cmd_list = []
    with open(jobs_stats_filepath, "w") as f:
        json.dump({"num_jobs": len(config_paths), "states": get_state_counter(job_paths)}, f, indent=4)
    for config_path, cfg, job_path in zip(config_paths, configs, job_paths):
        if OutputFolder(job_path).state == ExperimentState.FINISHED:
            continue
        if cmd == "uv":
            cmd_items = [
                "uv",
                "run",
                "--no-sync",
                script,
                str(config_path),
                "--root_output_folder",
                str(job_path.parent),
                "--output_folder",
                str(job_path.name),
            ]
        elif cmd == "sbatch":
            cmd_items = [
                "sbatch",
                "--job-name",
                f"{cfg.model.name}-{cfg.dataset_config.dataset_name}",
                "--output",
                str(job_path / "slurm-%j.log"),
                "--error",
                str(job_path / "slurm-%j.log"),
                f"{script} {config_path} {job_path.parent} {job_path.name}",
            ]
        else:
            raise ValueError(f"Invalid command: {cmd}")
        cmd_str = " ".join(cmd_items)
        cmd_list.append(cmd_str)
        click.echo(f"Submitting job with command:\n\t{cmd_str}\n")
        try:
            if all([not dry_run, out_file is None]):
                os.system(cmd_str)
        except Exception as e:
            click.echo(f"Error submitting job for config {config_path}: {e}")
        with open(jobs_stats_filepath, "w") as f:
            json.dump({"num_jobs": len(config_paths), "states": get_state_counter(job_paths)}, f, indent=4)
    if out_file is not None:
        out_filepath = Path(out_file)
        out_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(out_filepath, "w") as f:
            f.write("\n".join(cmd_list))
        click.echo(f"CMDs written to {out_filepath}")
    click.echo(f"Job states: {json.dumps(get_state_counter(job_paths), indent=4)}")
    click.echo(f"Finished launch_jobs_cli.")
    return 0

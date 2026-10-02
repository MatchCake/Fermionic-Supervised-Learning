import json
from pathlib import Path

import click

from ..configs.experiment_config import ExperimentConfig


@click.command()
@click.option(
    "--config_file",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment config YAML file.",
)
@click.option(
    "--output",
    type=click.Path(),
    default="workflow/hashes.json",
    show_default=True,
    help="Path to write the hashes JSON file.",
)
def generate_hashes_cli(config_file: str, output: str):
    """Generate workflow/hashes.json from an experiment config YAML."""
    config_path = Path(config_file)
    exp_config: ExperimentConfig = ExperimentConfig.from_yaml_file(config_path)  # type: ignore
    click.echo(f"Loaded config: {exp_config.experiment_name}")

    n_dataset_configs = len(exp_config.datasets.to_dataset_configs())
    n_model_configs = sum(len(model.to_model_configs()) for model in exp_config.models)
    full_grid_size = n_dataset_configs * n_model_configs

    all_configs = sorted(exp_config.to_single_experiment_configs(), key=lambda c: c.hash)
    n_skipped = full_grid_size - len(all_configs)
    if n_skipped > 0:
        click.echo(f"Skipped {n_skipped} infeasible config(s) (Pfaffian feature dimension over limit).")
    click.echo(f"Total configs: {len(all_configs)}")

    hashes_data = {
        "experiment_name": exp_config.experiment_name,
        "configs": [
            {
                "hash": cfg.hash,
                "mem_mb": cfg.model.slurm_resources["mem_mb"],
                "runtime_min": cfg.model.slurm_resources["runtime_min"],
            }
            for cfg in all_configs
        ],
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(hashes_data, indent=2))
    click.echo(f"Hashes written to {output_path} ({len(all_configs)} entries)")

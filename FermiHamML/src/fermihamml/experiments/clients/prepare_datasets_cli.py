import itertools
import os
from pathlib import Path

import click

from ...datasets import dataset_name_to_type_map, get_dataset_cls_by_name


@click.command()
@click.option(
    "--data_dir",
    type=click.Path(),
    envvar="DATA_DIR",
    default=Path(os.getcwd()) / "data",
)
def prepare_datasets_cli(data_dir: Path):
    """
    Prepare all datasets for the experiments.

    :param data_dir: Path to the root data directory where the datasets will be prepared.
        Defaults to the "data" folder in the current working directory or the value of the
        DATA_DIR environment variable.
    """
    click.echo(f"Preparing datasets...")
    for dataset_name, train in itertools.product(dataset_name_to_type_map, [True, False]):
        click.echo(f"Preparing dataset {dataset_name}:{'train' if train else 'test'} ...")
        dataset = get_dataset_cls_by_name(dataset_name)(data_dir=data_dir, train=train)
        dataset.prepare_data()
    click.echo(f"Finished preparing datasets.")
    return 0

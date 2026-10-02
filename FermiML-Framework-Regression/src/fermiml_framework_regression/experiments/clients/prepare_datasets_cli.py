import os
from pathlib import Path
from typing import Dict, List, Tuple

import click

from ..configs.experiment_config import ExperimentConfig


def _dataset_variants(exp_config: ExperimentConfig, all_configs: bool) -> List[Tuple[str, Dict]]:
    """The dataset instances to materialise.

    One per dataset name is enough to populate a download cache, because the cache is keyed by
    the data source rather than by the subsample a job takes. ``all_configs`` widens this to
    every distinct option set, which also acts as a pre-flight check that each cell in the grid
    constructs.

    :param exp_config: The experiment configuration.
    :param all_configs: Whether to cover every distinct option set rather than one per dataset.
    :return: ``(dataset name, extra options)`` pairs, deduplicated and order-stable.
    :rtype: List[Tuple[str, Dict]]
    """
    variants: List[Tuple[str, Dict]] = []
    seen = set()
    for dataset_config in exp_config.datasets.to_dataset_configs():
        key = (dataset_config.dataset_name, tuple(sorted(dataset_config.extra_options.items())))
        name_only = dataset_config.dataset_name
        if all_configs and key in seen:
            continue
        if not all_configs and name_only in seen:
            continue
        seen.add(key if all_configs else name_only)
        variants.append((dataset_config.dataset_name, dict(dataset_config.extra_options)))
    return variants


@click.command()
@click.option("--exp_config_file", type=click.Path(exists=True), envvar="EXP_CONFIG_FILE", default=None)
@click.option(
    "--cache_dir",
    type=click.Path(),
    default=None,
    help="Where scikit-learn caches downloaded data. Defaults to $SCIKIT_LEARN_DATA, else ./datasets.",
)
@click.option(
    "--all_configs",
    is_flag=True,
    default=False,
    help="Materialise every distinct dataset configuration, not just one per dataset.",
)
def prepare_datasets_cli(exp_config_file: Path, cache_dir: str, all_configs: bool) -> int:
    """Download and cache everything the sweep's datasets need, before any job is submitted.

    Compute nodes have no outbound internet, so a job that reaches for
    ``fetch_california_housing`` there fails. This populates the cache once, on a login node,
    into a directory the Snakefile binds into every container as ``/app/datasets``.

    It has to run **inside the Apptainer image**: the cluster virtualenv holds only Snakemake and
    the SLURM executor plugin, not scikit-learn or this package. See workflow/README.md for the
    invocation.

    :param exp_config_file: Path to the experiment config YAML.
    :param cache_dir: Directory scikit-learn should cache into.
    :param all_configs: Whether to cover every distinct dataset configuration.
    :return: 0 on success.
    """
    if exp_config_file is None:
        raise click.UsageError("Provide --exp_config_file or set EXP_CONFIG_FILE.")
    resolved_cache = cache_dir or os.environ.get("SCIKIT_LEARN_DATA") or "datasets"
    cache_path = Path(resolved_cache)
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ["SCIKIT_LEARN_DATA"] = str(cache_path.resolve())
    click.echo(f"Caching datasets into {cache_path.resolve()}")

    from ..single_experiment import SingleExperiment

    exp_config: ExperimentConfig = ExperimentConfig.from_yaml_file(Path(exp_config_file))  # type: ignore
    variants = _dataset_variants(exp_config, all_configs)
    click.echo(f"Preparing {len(variants)} dataset variant(s) from {exp_config.experiment_name} ...")

    failures = []
    for dataset_name, extra_options in variants:
        dataset_cls = next(
            (cls for cls in SingleExperiment.ALLOWED_DATASET_CLS if cls.DATASET_NAME.lower() == dataset_name.lower()),
            None,
        )
        if dataset_cls is None:
            failures.append((dataset_name, "not registered in SingleExperiment.ALLOWED_DATASET_CLS"))
            click.echo(f"  FAIL {dataset_name}: not a known dataset")
            continue
        try:
            dataset = dataset_cls(**extra_options)
            dataset.prepare_data()
            click.echo(
                f"  ok   {dataset_name} {extra_options or ''} -> "
                f"train={dataset.x_train.shape} val={dataset.x_val.shape} test={dataset.x_test.shape}"
            )
        except Exception as error:  # noqa: BLE001 -- report every dataset before failing
            failures.append((dataset_name, str(error)))
            click.echo(f"  FAIL {dataset_name} {extra_options or ''}: {error}")

    if failures:
        click.echo(f"\n{len(failures)} dataset(s) could not be prepared:")
        for dataset_name, reason in failures:
            click.echo(f"  {dataset_name}: {reason}")
        raise click.ClickException("dataset preparation incomplete; the sweep would fail on compute nodes")
    click.echo(f"\nAll datasets ready. Cache: {cache_path.resolve()}")
    return 0

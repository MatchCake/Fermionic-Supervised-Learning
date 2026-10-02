import itertools
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

from dataclass_wizard import YAMLWizard
from tqdm import tqdm

from .datasets_config import DatasetsConfig
from .model_hps_config import ModelHpsConfig
from .single_experiment_config import SingleExperimentConfig


@dataclass
class ExperimentConfig(YAMLWizard):
    """Top-level experiment configuration.

    :param experiment_name: Unique name for the experiment run; determines the
        output directory ``data/{experiment_name}/results/``.
    :param datasets: Dataset configurations with CV settings.
    :param models: Model configurations with hyperparameter ranges.
    """

    experiment_name: str
    datasets: DatasetsConfig
    models: List[ModelHpsConfig]

    def to_single_experiment_configs(self) -> List[SingleExperimentConfig]:
        """Expand into one :class:`SingleExperimentConfig` per (dataset, model) pair.

        :return: List of :class:`SingleExperimentConfig` instances.
        """
        dataset_configs = self.datasets.to_dataset_configs()
        model_configs = sum([m.to_model_configs() for m in self.models], [])
        return [
            SingleExperimentConfig(dataset=d, model=m) for d, m in itertools.product(dataset_configs, model_configs)
        ]

    def to_single_experiment_yamls(
        self,
        root_dir: Union[Path, str],
        rm_existing: bool = True,
    ) -> List[Path]:
        """Write one YAML file per :class:`SingleExperimentConfig`.

        :param root_dir: Root directory; files are written under
            ``{root_dir}/{experiment_name}/configs/``.
        :param rm_existing: Whether to remove the output directory first.
        :return: List of written YAML file paths.
        """
        root_dir = Path(root_dir)
        output_dir = root_dir / self.experiment_name / "configs"
        if rm_existing:
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        yaml_paths = []
        p_bar = tqdm(
            self.to_single_experiment_configs(),
            desc=f"Saving experiment configs for {self.experiment_name}",
        )
        for cfg in p_bar:
            yaml_path = output_dir / f"{cfg.hash}.yaml"
            cfg.to_yaml_file(yaml_path)
            yaml_paths.append(yaml_path)
        p_bar.close()
        return yaml_paths

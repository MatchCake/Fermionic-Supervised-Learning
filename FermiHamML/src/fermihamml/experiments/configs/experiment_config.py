import itertools
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from dataclass_wizard import YAMLWizard
from tqdm import tqdm

from .datasets_config import DatasetsConfig
from .model_config import ModelConfig
from .opt_config import OptConfig
from .single_experiment_config import SingleExperimentConfig


@dataclass
class ExperimentConfig(YAMLWizard):
    experiment_name: str
    datasets: DatasetsConfig
    models: List[ModelConfig]
    opt_config: OptConfig = field(default_factory=OptConfig)

    def to_single_experiment_configs(self) -> List[SingleExperimentConfig]:
        single_experiment_configs = []
        dataset_configs = self.datasets.to_dataset_configs()
        for dataset_config, model_config in itertools.product(dataset_configs, self.models):
            single_experiment_configs.append(
                SingleExperimentConfig(
                    dataset_config=dataset_config,
                    model=model_config,
                    opt_config=self.opt_config,
                )
            )
        return single_experiment_configs

    def to_single_experiment_yamls(
        self,
        root_dir: Union[Path, str],
        rm_existing: bool = True,
    ) -> List[Path]:
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
        for single_experiment_config in p_bar:
            yaml_path = output_dir / f"{single_experiment_config.hash}.yaml"
            single_experiment_config.to_yaml_file(yaml_path)
            yaml_paths.append(yaml_path)
        p_bar.close()
        return yaml_paths

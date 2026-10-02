import datetime
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import click
from lightning.fabric.utilities.warnings import PossibleUserWarning
from matchcake_opt.datasets import (
    dataset_name_to_type_map as mc_opt_dataset_name_to_type_map,
)
from matchcake_opt.tr_pipeline.automl_pipeline import AutoMLPipeline
from matchcake_opt.tr_pipeline.lightning_pipeline import LightningPipeline

from ..datasets import dataset_name_to_type_map
from ..datasets.collated_datamodule import CollatedDataModule
from ..models.nif_ising_cls import NIFIsingCls
from ..utils.output_folder import ExperimentState, OutputFolder
from .configs.single_experiment_config import SingleExperimentConfig


class SingleExperiment:
    ALLOWED_MODEL_CLS = [
        NIFIsingCls,
    ]
    SCRIPT = "single_exp"

    def __init__(
        self,
        config: SingleExperimentConfig,
        root_output_folder: Optional[Union[str, Path]] = None,
        output_folder: Optional[Union[str, Path]] = None,
        log_func: Callable[[str], None] = click.echo,
        raise_errors: bool = False,
    ):
        self._config = config
        if root_output_folder is None:
            root_output_folder = Path(os.getcwd()) / "data"
        self._root_output_folder: Path = Path(root_output_folder)
        if output_folder is None:
            output_folder: Path = Path(self._config.hash)  # type: ignore
        self._output_folder: OutputFolder = OutputFolder(self._root_output_folder / output_folder)  # type: ignore
        self._config.to_yaml_file(self.output_folder.path / "config.yaml")  # type: ignore
        self.log_func = log_func
        self.raise_errors = raise_errors
        self._pipeline: Union[AutoMLPipeline, LightningPipeline] = self._build_pipeline()

    def run(self) -> int:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=PossibleUserWarning)
        start_time = time.time()
        self.output_folder.state = ExperimentState.RUNNING
        self.log_func(
            f"METADATA:\n"
            f"{json.dumps(self.output_folder.metadata_file.raveled_state, indent=2, sort_keys=True, default=str)}"
        )
        self.output_folder.update_data(**self.config.to_dict())
        try:
            self._run_pipeline()
            self.output_folder.state = ExperimentState.FINISHED
        except Exception as e:
            self.output_folder.state = ExperimentState.FAILED
            self.output_folder.state_file.write(str(e))
            if self.raise_errors:
                raise e
        self.log_func(
            f"Output:\n"
            f"{json.dumps(self.output_folder.data_file.raveled_state, indent=2, sort_keys=True, default=str)}"
        )
        end_time = time.time()
        # time in dd:hh:mm:ss format
        fmt_elapsed_time = str(datetime.timedelta(seconds=int(end_time - start_time)))
        self.output_folder.update_metadata(run_time=fmt_elapsed_time)
        return 0

    def _build_pipeline(self) -> Union[AutoMLPipeline, LightningPipeline]:
        mc_opt_dataset_name_to_type_map.update(dataset_name_to_type_map)
        datamodule = CollatedDataModule.from_dataset_name(
            dataset_name=self.config.dataset_config.dataset_name,
            split_id=self.config.dataset_config.split_id,
            batch_size=self.config.dataset_config.batch_size,
            random_state=self.config.dataset_config.seed,
            num_workers=1,
        )
        model_cls = self._get_model_cls()
        if self.config.opt_config.automl_iterations == 0:
            pipeline = LightningPipeline(
                model_cls=model_cls,
                datamodule=datamodule,
                checkpoint_folder=self.output_folder,  # type: ignore
                max_epochs=self.config.opt_config.max_epochs,
                max_time=self.config.opt_config.max_time,
                monitor=self.config.opt_config.monitor,
                accelerator=self.config.opt_config.accelerator,
                verbose=True,
                overwrite_fit=self.config.opt_config.automl_overwrite_fit,
                **self.config.model.extra_options,
            )
        else:
            pipeline = AutoMLPipeline(  # type: ignore
                model_cls=model_cls,
                datamodule=datamodule,
                checkpoint_folder=self.output_folder,  # type: ignore
                automl_iterations=self.config.opt_config.automl_iterations,
                inner_max_epochs=self.config.opt_config.inner_max_epochs,
                inner_max_time=self.config.opt_config.inner_max_time,
                max_epochs=self.config.opt_config.max_epochs,
                max_time=self.config.opt_config.max_time,
                monitor=self.config.opt_config.monitor,
                automl_overwrite_fit=self.config.opt_config.automl_overwrite_fit,
                accelerator=self.config.opt_config.accelerator,
                model_args=self.config.model.extra_options,
                **self.config.opt_config.extra_options,
            )
        return pipeline

    def _run_pipeline(self):
        if isinstance(self._pipeline, AutoMLPipeline):
            return self._run_automl_pipline()
        elif isinstance(self._pipeline, LightningPipeline):
            return self._run_lightning_pipeline()
        raise ValueError(f"Pipeline {self._pipeline} is not supported")  # pragma: no cover

    def _run_automl_pipline(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._pipeline.run()
        self.output_folder.update_data({"best_hparams": self._pipeline.get_best_pipeline_params()})  # type: ignore
        lt_pipeline, val_metrics = self._pipeline.run_best_pipeline()  # type: ignore
        # TODO: Find a way to add the number of parameters
        # self.output_folder.update_data({"num_params": self._pipeline.model.num_params})  # type: ignore
        self.output_folder.update_data(val_metrics)
        try:
            test_metrics = lt_pipeline.run_test(ckpt_path="best")
        except ValueError:  # pragma: no cover
            test_metrics = lt_pipeline.run_test(ckpt_path="last")
        self.output_folder.update_data(test_metrics)
        return val_metrics, test_metrics

    def _run_lightning_pipeline(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        val_metrics = self._pipeline.run()
        self.output_folder.update_data({"hparams": self._pipeline.model.hparams})  # type: ignore
        self.output_folder.update_data({"num_params": self._pipeline.model.num_params})  # type: ignore
        self.output_folder.update_data(val_metrics)
        try:
            test_metrics = self._pipeline.run_test(ckpt_path="best")  # type: ignore
        except ValueError:  # pragma: no cover
            test_metrics = self._pipeline.run_test(ckpt_path="last")  # type: ignore
        self.output_folder.update_data(test_metrics)
        return val_metrics, test_metrics

    def _get_model_cls(self):
        for model_cls in self.ALLOWED_MODEL_CLS:
            if self.config.model.name.lower() == model_cls.MODEL_NAME.lower():
                return model_cls
        raise ValueError(f"Model {self.config.model.name.lower()} is not supported")  # pragma: no cover

    @property
    def config(self):
        return self._config

    @property
    def output_folder(self) -> OutputFolder:
        return self._output_folder

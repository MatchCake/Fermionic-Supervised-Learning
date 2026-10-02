import datetime
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch

try:
    import psutil as _psutil

    _PSUTIL_PROC = _psutil.Process()

    def _mem_mb() -> float:
        return _PSUTIL_PROC.memory_info().rss / 1024**2

except ImportError:

    def _mem_mb() -> float:
        try:
            with open("/proc/self/status") as _f:
                for _line in _f:
                    if _line.startswith("VmRSS:"):
                        return float(_line.split()[1]) / 1024
        except Exception:
            pass
        return float("nan")


import traceback

import click
from lightning.fabric.utilities.warnings import PossibleUserWarning
from pythonbasictools import OutputFolder
from pythonbasictools.experiment_utils.output_folder import ExperimentState
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..datasets.bars_and_stripes import BarsAndStripes
from ..datasets.downscaled_mnist import DownscaledMnist
from ..datasets.hidden_manifold import HiddenManifold
from ..datasets.hidden_manifold_diff import HiddenManifoldDiff
from ..datasets.linearly_separable import LinearlySeparable
from ..models.fermi_angle_mpf_kernel_classifier import FermiAngleMpfKernelClassifier
from ..models.fermi_iqp_kernel_classifier import FermiIQPKernelClassifier
from ..models.fermi_kitchen_sink_kernel_classifier import (
    FermiKitchenSinkKernelClassifier,
)
from ..models.fermi_vanilla_kernel_classifier import FermiVanillaKernelClassifier
from ..models.fermi_vanilla_mpf_kernel_classifier import FermiVanillaMpfKernelClassifier
from ..models.fermi_vanilla_mpf_plus_kernel_classifier import (
    FermiVanillaMpfPlusKernelClassifier,
)
from ..models.fermi_vanilla_plus_kernel_classifier import (
    FermiVanillaPlusKernelClassifier,
)
from ..models.iqp_kernel_classifier import IQPKernelClassifier
from ..models.kitchen_sink_kernel_classifier import KitchenSinkKernelClassifier
from ..utils.threading_config import configure_threads, thread_report
from .configs.single_experiment_config import SingleExperimentConfig


class SingleExperiment:
    ALLOWED_DATASET_CLS = [
        BarsAndStripes,
        HiddenManifold,
        LinearlySeparable,
        HiddenManifoldDiff,
        DownscaledMnist,
    ]
    ALLOWED_MODEL_CLS = [
        IQPKernelClassifier,
        FermiIQPKernelClassifier,
        FermiVanillaKernelClassifier,
        FermiVanillaPlusKernelClassifier,
        FermiVanillaMpfKernelClassifier,
        FermiVanillaMpfPlusKernelClassifier,
        FermiAngleMpfKernelClassifier,
        KitchenSinkKernelClassifier,
        FermiKitchenSinkKernelClassifier,
    ]
    SCRIPT = "single_exp"
    BASE_RANDOM_STATE = 0

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
        self._model_cls = self._get_model_cls()
        self._dataset_cls = self._get_dataset_cls()

    def run(self) -> int:
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=PossibleUserWarning)
        n_threads = configure_threads()
        self.log_func(f"[THREADS] using {n_threads} core(s); report={json.dumps(thread_report(), default=str)}")
        start_time = time.time()
        self.output_folder.unfreeze()
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
            self.log_func(f"[ERROR] Pipeline failed:\n{traceback.format_exc()}")
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
        self.output_folder.data_file.save()
        self.output_folder.metadata_file.save()
        self.output_folder.freeze()
        return 0

    def _log_mem(self, tag: str) -> None:
        mb = _mem_mb()
        try:
            import psutil as _ps

            vms_mb = _ps.Process().memory_info().vms / 1024**2
            self.log_func(f"[MEM] {tag}: rss={mb:.0f} MB  vms={vms_mb:.0f} MB")
        except Exception:
            self.log_func(f"[MEM] {tag}: rss={mb:.0f} MB")
        self.output_folder.update_data(**{f"mem_mb/{tag}": mb})
        self.output_folder.data_file.save()

    def _seed_everything(self) -> int:
        """Seed every random source this run touches and return the seed used.

        The seed is derived from the split so that each split draws differently while staying reproducible.
        Three sources need it: torch's global generator, which initialises the ``LazyLinear`` encoder of the
        matchgate kernels; the legacy numpy generator, which backs the sampling of matchcake's non-interacting
        fermionic device; and the stdlib generator. The quantum-kitchen-sink devices are seeded separately,
        through the ``random_state`` handed to the model in :meth:`_run_pipeline`.

        This derives the seed in code rather than in the config, so the experiment hash is unaffected.

        :return: the seed applied to every generator.
        :rtype: int
        """
        seed = self.BASE_RANDOM_STATE + self.config.dataset.split_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return seed

    def _run_pipeline(self):
        t0 = time.time()
        self._log_mem("pipeline_start")
        seed = self._seed_everything()
        self.log_func(f"[SEED] all generators seeded with {seed}")
        self.output_folder.update_metadata(random_state=seed)

        dataset = self._dataset_cls(
            cv_split_id=self.config.dataset.split_id,
            cv_splits=self.config.dataset.cv_splits,
            val_size=self.config.dataset.val_size,
            cv_random_state=self.config.dataset.splits_seed,
            **self.config.dataset.extra_options,
        )
        dataset.prepare_data()
        self._log_mem("after_prepare_data")
        self.log_func(f"[DATA] train={dataset.x_train.shape}  val={dataset.x_val.shape}  test={dataset.x_test.shape}")

        model_kwargs = {**self.config.model.hyperparameters, **self.config.model.extra_options}
        # Passed outside the hyperparameters dict so it never enters the config, and hence never the hash.
        model_kwargs.setdefault("random_state", seed)
        model = self._model_cls(**model_kwargs)
        self._log_mem("after_model_init")

        self.log_func(f"[INFO] Starting model.fit on {dataset.x_train.shape} samples...")
        t_fit = time.time()
        model.fit(dataset.x_train, dataset.y_train)
        self.log_func(f"[TIME] fit: {time.time() - t_fit:.1f}s")
        self._log_mem("after_fit")

        self.log_func(f"[INFO] fit complete. Starting predictions...")
        splits = {
            "train": (dataset.x_train, dataset.y_train),
            "val": (dataset.x_val, dataset.y_val),
            "test": (dataset.x_test, dataset.y_test),
        }
        metrics = {}
        for split, (x, y_true) in splits.items():
            t_pred = time.time()
            y_pred = model.predict(x)
            self.log_func(f"[TIME] predict/{split}: {time.time() - t_pred:.1f}s")
            self._log_mem(f"after_predict/{split}")
            metrics[f"{split}_accuracy"] = accuracy_score(y_true, y_pred)
            metrics[f"{split}_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics[f"{split}_recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics[f"{split}_precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)

        self.log_func(f"[TIME] pipeline_total: {time.time() - t0:.1f}s")
        self.output_folder.update_data(**metrics)
        self.output_folder.data_file.save()
        return metrics

    def _get_model_cls(self):
        for model_cls in self.ALLOWED_MODEL_CLS:
            if self.config.model.name.lower() == model_cls.MODEL_NAME.lower():
                return model_cls
        raise ValueError(f"Model {self.config.model.name.lower()} is not supported")  # pragma: no cover

    def _get_dataset_cls(self):
        for dataset_cls in self.ALLOWED_DATASET_CLS:
            if self.config.dataset.dataset_name.lower() == dataset_cls.DATASET_NAME.lower():
                return dataset_cls
        raise ValueError(f"Dataset {self.config.dataset.dataset_name.lower()} is not supported")

    @property
    def config(self):
        return self._config

    @property
    def output_folder(self) -> OutputFolder:
        return self._output_folder

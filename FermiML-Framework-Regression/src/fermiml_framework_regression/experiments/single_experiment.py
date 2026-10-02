import datetime
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import click
import numpy as np
from pythonbasictools import OutputFolder
from pythonbasictools.experiment_utils.output_folder import ExperimentState
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..datasets.california_housing import CaliforniaHousingDataset
from ..datasets.dataset import Dataset
from ..datasets.friedman import FriedmanDataset
from ..datasets.magnetisation import MagnetisationDataset
from ..models.angle_mpf_plus_gp_regressor import AngleMpfPlusGPRegressor
from ..models.angle_nif_gp_regressor import AngleNIFGPRegressor
from ..models.classical_gp_regressor import ClassicalGPRegressor
from ..models.classical_svr import ClassicalSVR
from ..models.vanilla_mpf_plus_gp_regressor import VanillaMpfPlusGPRegressor
from ..models.vanilla_nif_gp_regressor import VanillaNIFGPRegressor
from ..models.vanilla_nif_kernel_regressor import VanillaNIFKernelRegressor
from .configs.single_experiment_config import SingleExperimentConfig


class SingleExperiment:
    """Orchestrates a single (dataset, model, CV split) experiment.

    Loads the dataset, trains the model, evaluates on train/val/test splits,
    and persists results under ``data/{experiment_name}/results/{hash}/``.

    :param config: Experiment configuration.
    :param root_output_folder: Root directory for results (default: ``data/``).
    :param output_folder: Subdirectory name within the results directory
        (default: the config hash).
    :param log_func: Callable for logging; defaults to :func:`click.echo`.
    :param raise_errors: Whether to re-raise exceptions instead of logging them.
    """

    # Annotated rather than inferred: without it mypy widens the literal to `list[type]`
    # and every DATASET_NAME lookup against it becomes an attribute error.
    ALLOWED_DATASET_CLS: List[Type[Dataset]] = [
        FriedmanDataset,
        CaliforniaHousingDataset,
        MagnetisationDataset,
    ]
    ALLOWED_MODEL_CLS = [
        ClassicalGPRegressor,
        ClassicalSVR,
        VanillaNIFKernelRegressor,
        AngleNIFGPRegressor,
        VanillaNIFGPRegressor,
        AngleMpfPlusGPRegressor,
        VanillaMpfPlusGPRegressor,
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
            output_folder = Path(self._config.hash)  # type: ignore
        self._output_folder: OutputFolder = OutputFolder(self._root_output_folder / output_folder)  # type: ignore
        self._config.to_yaml_file(self.output_folder.path / "config.yaml")  # type: ignore
        self.log_func = log_func
        self.raise_errors = raise_errors
        self._model_cls = self._get_model_cls()
        self._dataset_cls = self._get_dataset_cls()

    def run(self) -> int:
        """Execute the experiment and persist results.

        :return: 0 on success.
        """
        warnings.filterwarnings("ignore", category=FutureWarning)
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
            if self.raise_errors:
                raise e
        self.log_func(
            f"Output:\n{json.dumps(self.output_folder.data_file.raveled_state, indent=2, sort_keys=True, default=str)}"
        )
        end_time = time.time()
        fmt_elapsed = str(datetime.timedelta(seconds=int(end_time - start_time)))
        self.output_folder.update_metadata(run_time=fmt_elapsed)
        self.output_folder.data_file.save()
        self.output_folder.metadata_file.save()
        self.output_folder.freeze()
        return 0

    def seed_everything(self) -> int:
        """Seed every global RNG a job can draw from, deterministically per configuration.

        The models seed what they own, but a benchmark addressed by config hash cannot rely on
        that alone: a model that quietly reaches for a global RNG produces a different result on
        a requeue and overwrites its own cached answer with a different number. Seeding here
        makes the whole pipeline reproducible regardless of what a model does internally.

        The seed is derived from the configuration hash rather than fixed, so two different jobs
        do not share an RNG stream while each job stays reproducible on its own.

        :return: The seed used.
        :rtype: int
        """
        import random

        seed = int(self.config.hash[:8], 16)
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch

            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as error:  # noqa: BLE001 -- seeding must not fail the run
            self.log_func(f"torch seeding unavailable: {error}")
        return seed

    def _run_pipeline(self) -> None:
        seed = self.seed_everything()
        self.output_folder.update_data(run_seed=seed)
        dataset = self._dataset_cls(
            cv_split_id=self.config.dataset.split_id,
            cv_splits=self.config.dataset.cv_splits,
            val_size=self.config.dataset.val_size,
            cv_random_state=self.config.dataset.splits_seed,
            **self.config.dataset.extra_options,
        )
        dataset.prepare_data()
        model = self._model_cls(**self.config.model.hyperparameters, **self.config.model.extra_options)
        fit_start = time.time()
        model.fit(dataset.x_train, dataset.y_train)
        fit_seconds = time.time() - fit_start

        splits = {
            "train": (dataset.x_train, dataset.y_train),
            "val": (dataset.x_val, dataset.y_val),
            "test": (dataset.x_test, dataset.y_test),
        }
        metrics: Dict[str, Any] = {"fit_seconds": float(fit_seconds)}
        predictions = {}
        for split_name, (x, y_true) in splits.items():
            predict_start = time.time()
            y_pred = model.predict(x)
            metrics[f"{split_name}_predict_seconds"] = float(time.time() - predict_start)
            predictions[split_name] = y_pred
            mse = mean_squared_error(y_true, y_pred)
            metrics[f"{split_name}_mse"] = float(mse)
            metrics[f"{split_name}_rmse"] = float(mse**0.5)
            metrics[f"{split_name}_r2"] = float(r2_score(y_true, y_pred))
            metrics[f"{split_name}_mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics[f"n_{split_name}"] = int(len(y_true))
            metrics[f"{split_name}_y_mean"] = float(np.mean(y_true))
            metrics[f"{split_name}_y_std"] = float(np.std(y_true))
        metrics["n_features"] = int(dataset.x_train.shape[-1])
        metrics["test_targets"] = [float(value) for value in dataset.y_test]
        metrics["test_predictions"] = [float(value) for value in predictions["test"]]
        metrics["val_targets"] = [float(value) for value in dataset.y_val]
        metrics["val_predictions"] = [float(value) for value in predictions["val"]]
        if hasattr(model, "predict_std"):
            try:
                test_std = np.asarray(model.predict_std(dataset.x_test), dtype=float)
                metrics["test_predictive_std"] = [float(value) for value in test_std]
                metrics["test_predictive_std_mean"] = float(np.mean(test_std))
            except Exception as error:  # noqa: BLE001 -- a diagnostic must not fail the run
                self.log_func(f"predictive std unavailable: {error}")
        metrics.update(self._model_diagnostics(model))
        metrics.update(self._provenance())
        self.output_folder.update_data(**metrics)
        self.output_folder.data_file.save()

    def _provenance(self) -> Dict[str, Any]:
        """Record peak memory and the versions the run was produced with.

        `run_time` in the metadata has one-second resolution, which is useless for jobs that
        finish in tenths of a second, and it says nothing about memory. Peak RSS is what a
        future provisioning pass needs, and the package versions matter because the kernel comes
        from a git-pinned matchcake: a result is only interpretable next to the commit that
        produced it.

        :return: The provenance fields that could be read.
        :rtype: Dict[str, Any]
        """
        provenance: Dict[str, Any] = {}
        try:
            import resource

            # ru_maxrss is kilobytes on Linux.
            provenance["peak_rss_mb"] = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
        except Exception as error:  # noqa: BLE001 -- a diagnostic must not fail the run
            self.log_func(f"peak RSS unavailable: {error}")
        try:
            from importlib.metadata import version

            for package in ("FermiML-Framework-Regression", "matchcake", "torchpfaffian", "pennylane"):
                provenance[f"version_{package.replace('-', '_').lower()}"] = version(package)
        except Exception as error:  # noqa: BLE001 -- a diagnostic must not fail the run
            self.log_func(f"version provenance unavailable: {error}")
        return provenance

    def _model_diagnostics(self, model: Any) -> Dict[str, Any]:
        """Record what the fitted model resolved, for the cost and scaling figures.

        With ``n_qubits: -1`` throughout the grid, the qubit count and the Pfaffian feature-map
        dimension are decided at fit time and are not recoverable from the config alone. The
        Gram statistics are what a kernel-concentration figure needs, and recomputing them here
        costs one forward pass on an already-fitted kernel.

        Every probe is optional and individually guarded: a diagnostic that fails must never
        turn a job that produced valid metrics into a failed one.

        :param model: The fitted model.
        :return: Whatever could be measured, as a flat mapping.
        :rtype: Dict[str, Any]
        """
        diagnostics: Dict[str, Any] = {}
        for attribute in ("effective_alpha_", "n_jitter_retries_"):
            value = getattr(model, attribute, None)
            if value is not None:
                diagnostics[attribute.rstrip("_")] = float(value)
        kernel = getattr(model, "nif_kernel_", model)
        for attribute in ("n_qubits", "sector_m", "n_terms"):
            try:
                value = getattr(kernel, attribute, None)
                if value is not None:
                    diagnostics[f"resolved_{attribute}"] = int(value)
            except Exception as error:  # noqa: BLE001 -- a diagnostic must not fail the run
                self.log_func(f"diagnostic {attribute!r} unavailable: {error}")
        gp = getattr(model, "gp_", None)
        fitted_kernel = getattr(gp, "kernel_", None)
        if fitted_kernel is not None:
            diagnostics["fitted_kernel"] = str(fitted_kernel)
        try:
            gram = self._train_gram(model, kernel)
            if gram is not None:
                off_diagonal = gram[~np.eye(len(gram), dtype=bool)]
                diagnostics["gram_diagonal_mean"] = float(np.mean(np.diag(gram)))
                diagnostics["gram_offdiagonal_abs_mean"] = float(np.mean(np.abs(off_diagonal)))
                diagnostics["gram_offdiagonal_abs_max"] = float(np.max(np.abs(off_diagonal)))
        except Exception as error:  # noqa: BLE001 -- a diagnostic must not fail the run
            self.log_func(f"Gram diagnostics unavailable: {error}")
        return diagnostics

    def _train_gram(self, model: Any, kernel: Any) -> Optional[np.ndarray]:
        """The training Gram matrix of a kernel model, or None for a model without one.

        Computed the way each model computes it: the covariance kernels through
        ``compute_similarities`` (which bypasses ``GramMatrix`` and keeps the true, non-unit
        diagonal), the fidelity model through its own ``transform`` on scaled inputs, since that
        is what its SVR was actually fitted on.

        :param model: The fitted model.
        :param kernel: The kernel carried by that model.
        :return: The Gram matrix, or None when the model has no kernel.
        :rtype: Optional[np.ndarray]
        """
        from matchcake.utils.torch_utils import to_numpy

        x_train = getattr(kernel, "x_train_", None)
        if x_train is None:
            return None
        if hasattr(kernel, "compute_similarities") and hasattr(kernel, "feature_map"):
            return to_numpy(kernel.compute_similarities(x_train, x_train))
        if hasattr(model, "transform"):
            return to_numpy(model.transform(to_numpy(x_train)))
        return None

    def _get_model_cls(self):
        for model_cls in self.ALLOWED_MODEL_CLS:
            if self.config.model.name.lower() == model_cls.MODEL_NAME.lower():
                return model_cls
        raise ValueError(f"Model {self.config.model.name!r} is not supported")  # pragma: no cover

    def _get_dataset_cls(self):
        for dataset_cls in self.ALLOWED_DATASET_CLS:
            if self.config.dataset.dataset_name.lower() == dataset_cls.DATASET_NAME.lower():
                return dataset_cls
        raise ValueError(f"Dataset {self.config.dataset.dataset_name!r} is not supported")

    @property
    def config(self) -> SingleExperimentConfig:
        return self._config

    @property
    def output_folder(self) -> OutputFolder:
        return self._output_folder

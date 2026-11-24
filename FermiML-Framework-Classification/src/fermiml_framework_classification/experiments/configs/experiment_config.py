import itertools
import shutil
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import List, Optional, Union

from dataclass_wizard import YAMLWizard
from tqdm import tqdm

from .dataset_config import DatasetConfig
from .datasets_config import DatasetsConfig
from .model_config import ModelConfig
from .model_hps_config import ModelHpsConfig
from .single_experiment_config import SingleExperimentConfig


@dataclass
class ExperimentConfig(YAMLWizard):
    experiment_name: str
    datasets: DatasetsConfig
    models: List[ModelHpsConfig]

    @staticmethod
    def _dataset_feature_dimension(dataset_config: DatasetConfig) -> Optional[int]:
        """The input dimension a dataset config will produce, or ``None`` for an unknown dataset name.

        :param dataset_config: The dataset configuration to resolve.
        :return: The number of input features, or ``None`` when the name matches no known dataset.
        :rtype: Optional[int]
        """
        from ...datasets.bars_and_stripes import BarsAndStripes
        from ...datasets.downscaled_mnist import DownscaledMnist
        from ...datasets.hidden_manifold import HiddenManifold
        from ...datasets.hidden_manifold_diff import HiddenManifoldDiff
        from ...datasets.linearly_separable import LinearlySeparable

        for dataset_cls in (BarsAndStripes, HiddenManifold, LinearlySeparable, HiddenManifoldDiff, DownscaledMnist):
            if dataset_config.dataset_name.lower() == dataset_cls.DATASET_NAME.lower():
                return dataset_cls.feature_dimension(dataset_config.extra_options)
        return None

    @classmethod
    def _model_config_is_feasible(cls, model_config: ModelConfig, dataset_config: DatasetConfig) -> bool:
        """Return whether a model config can run on a dataset config without a guaranteed failure.

        Pfaffian models (``VanillaMpf``, ``VanillaMpf+``, ``AngleMpf``) build a feature map of dimension
        :math:`\\binom{2n}{m}` over the :math:`2n` Majorana indices, and two things can make a combination
        impossible before it ever runs:

        - no non-trivial :math:`m`-subset exists, i.e. :math:`m > 2n - 2`, which raises ``ValueError`` in
          ``fit``. This is what a two-dimensional dataset does to ``sector_m=4``: :math:`n=2` leaves
          :math:`2n-2=2`;
        - the feature dimension exceeds ``MAX_PFAFFIAN_FEATURE_TERMS``, which raises ``MemoryError``.

        Both are checked against the qubit count the model will actually use, so a negative or missing
        ``n_qubits`` is resolved from the dataset's input dimension (rounded up to an even number, as the
        models do) rather than waved through.

        ``sectors`` spans several Majorana orders at once and is checked order by order, with the dimensions
        summed. Only ``VanillaMpf+`` accepts it: the odd orders it admits are carried by the displacement
        vector, which vanishes on the vacuum state the other Pfaffian models start from.

        :param model_config: The model configuration to check.
        :param dataset_config: The dataset configuration it would run on.
        :return: ``True`` if the config should be submitted, ``False`` otherwise.
        :rtype: bool
        """
        hyperparameters = model_config.hyperparameters
        n_qubits = hyperparameters.get("n_qubits")
        if model_config.name.lower() in ("pqc", "fpqc"):
            if n_qubits == "double":  # one wire pair per feature, a layout only the matchgate model has
                return model_config.name.lower() == "fpqc"
            return cls._pqc_qubit_count_is_distinct(n_qubits, dataset_config)
        sectors = hyperparameters.get("sectors")
        if sectors is None:
            sector_m = hyperparameters.get("sector_m")
            if sector_m is None:
                return True
            sectors = [sector_m]
        if n_qubits is None or n_qubits < 0:
            n_features = cls._dataset_feature_dimension(dataset_config)
            if n_features is None:  # unknown dataset: leave the decision to the job itself
                return True
            n_qubits = n_features + (n_features % 2)
        # An odd order is closed with the parity index of the extended algebra, so its submatrix is one wider
        # than the order itself and it only needs m <= 2n. Applying the even bound to it would drop feasible
        # points, notably every order-three combination on the two-dimensional configurations.
        for order in sectors:
            if order > (2 * n_qubits if order % 2 else 2 * n_qubits - 2):
                return False
        from ...models.fermi_vanilla_mpf_kernel_classifier import (
            FermiVanillaMpfKernelClassifier,
        )

        max_terms = FermiVanillaMpfKernelClassifier.MAX_PFAFFIAN_FEATURE_TERMS
        return sum(comb(2 * n_qubits, order) for order in sectors) <= max_terms

    @classmethod
    def _pqc_qubit_count_is_distinct(cls, n_qubits: Optional[Union[int, str]], dataset_config: DatasetConfig) -> bool:
        """Return whether a PQC or fPQC qubit setting gives a circuit that a larger setting does not already give.

        :param n_qubits: The requested qubit setting.
        :param dataset_config: The dataset configuration it would run on.
        :return: ``True`` if the config should be submitted, ``False`` otherwise.
        :rtype: bool
        """
        if n_qubits is None or (not isinstance(n_qubits, str) and n_qubits < 0) or n_qubits == "full":
            return True
        n_features = cls._dataset_feature_dimension(dataset_config)
        if n_features is None:  # unknown dataset: leave the decision to the job itself
            return True
        from ...models.pqc_encoding import PQCEncoding

        resolved = PQCEncoding.resolve_n_qubits(n_qubits, n_features)
        if not isinstance(n_qubits, str):
            return resolved < PQCEncoding.resolve_n_qubits("full", n_features)
        larger_rules = [
            rule for rule, divisor in PQCEncoding.QUBIT_RULES.items() if divisor < PQCEncoding.QUBIT_RULES[n_qubits]
        ]
        return all(PQCEncoding.resolve_n_qubits(rule, n_features) != resolved for rule in larger_rules)

    def to_single_experiment_configs(self) -> List[SingleExperimentConfig]:
        single_experiment_configs = []
        dataset_configs = self.datasets.to_dataset_configs()
        model_configs = sum([model_config.to_model_configs() for model_config in self.models], [])
        for dataset_config, model_config in itertools.product(dataset_configs, model_configs):
            if not self._model_config_is_feasible(model_config, dataset_config):
                continue
            single_experiment_configs.append(
                SingleExperimentConfig(
                    dataset=dataset_config,
                    model=model_config,
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

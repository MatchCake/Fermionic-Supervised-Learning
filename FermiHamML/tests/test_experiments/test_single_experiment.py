import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, create_autospec

import pytest
import torch
from matchcake_opt.tr_pipeline.automl_pipeline import AutoMLPipeline
from matchcake_opt.tr_pipeline.lightning_pipeline import LightningPipeline

from fermihamml.experiments.configs.dataset_config import DatasetConfig
from fermihamml.experiments.configs.model_config import ModelConfig
from fermihamml.experiments.configs.opt_config import OptConfig
from fermihamml.experiments.configs.single_experiment_config import (
    SingleExperimentConfig,
)
from fermihamml.experiments.single_experiment import SingleExperiment
from fermihamml.utils.output_folder import OutputFolder


class TestSingleExperiment:
    @pytest.fixture
    def config_with_automl(self):
        return SingleExperimentConfig(
            model=ModelConfig(
                name="NIFIsingCls",
                extra_options={"test_option": 0},
            ),
            dataset_config=DatasetConfig(
                dataset_name="Cifar2_mock",
                split_id=0,
                seed=0,
                batch_size=32,
            ),
            opt_config=OptConfig(
                automl_iterations=1,
                inner_max_epochs=1,
                max_epochs=1,
                automl_overwrite_fit=True,
            ),
        )

    @pytest.fixture
    def config_with_lightning(self):
        return SingleExperimentConfig(
            model=ModelConfig(
                name="NIFIsingCls",
                extra_options={"test_option": 0},
            ),
            dataset_config=DatasetConfig(
                dataset_name="Cifar2_mock",
                split_id=0,
                seed=0,
                batch_size=32,
            ),
            opt_config=OptConfig(
                automl_iterations=0,
                inner_max_epochs=1,
                max_epochs=1,
                automl_overwrite_fit=True,
            ),
        )

    @pytest.fixture
    def instance_with_automl(self, config_with_automl):
        return SingleExperiment(
            config=config_with_automl,
            root_output_folder=".tmp/single_experiment_test/test_root_automl",
        )

    @pytest.fixture
    def instance_with_lightning(self, config_with_lightning):
        return SingleExperiment(
            config=config_with_lightning,
            root_output_folder=".tmp/single_experiment_test/test_root_lightning",
        )

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        os.makedirs(".tmp/single_experiment_test", exist_ok=True)
        yield
        shutil.rmtree(".tmp/single_experiment_test", ignore_errors=True)

    @pytest.fixture(autouse=True)
    def cifar10_mock(self, monkeypatch):
        cifar10_cls_mock = MagicMock()
        mock = MagicMock()
        cifar10_cls_mock.return_value = mock
        mock.__getitem__.return_value = (torch.zeros(28, 28), torch.zeros(1).long())
        mock.__len__.return_value = 10
        monkeypatch.setattr(
            "fermihamml.experiments.single_experiment.dataset_name_to_type_map",
            {"Cifar2_mock".lower(): cifar10_cls_mock},
        )
        return mock

    @pytest.fixture(autouse=True)
    def mock_lightning_pipeline(self, monkeypatch):
        lightning_pipeline_cls_mock = LightningPipeline
        lightning_pipeline_instance_mock = create_autospec(LightningPipeline)
        lightning_pipeline_instance_mock.run.return_value = {"val_accuracy": 0.9}
        lightning_pipeline_cls_mock.__new__ = MagicMock(return_value=lightning_pipeline_instance_mock)
        monkeypatch.setattr("fermihamml.experiments.single_experiment.LightningPipeline", lightning_pipeline_cls_mock)
        return lightning_pipeline_instance_mock

    @pytest.fixture(autouse=True)
    def mock_automl_pipeline(self, monkeypatch):
        automl_pipeline_cls_mock = AutoMLPipeline
        automl_pipeline_instance_mock = create_autospec(AutoMLPipeline)
        automl_pipeline_instance_mock.run.return_value = None
        automl_pipeline_instance_mock.get_best_pipeline_params.return_value = {"param1": 0.1}
        best_lightning_pipeline_mock = MagicMock()
        best_lightning_pipeline_mock.run.return_value = {"val_accuracy": 0.95}
        best_lightning_pipeline_mock.run_test.return_value = {"test_accuracy": 0.92}
        automl_pipeline_instance_mock.run_best_pipeline.return_value = (
            best_lightning_pipeline_mock,
            {"val_accuracy": 0.95},
        )
        automl_pipeline_cls_mock.__new__ = MagicMock(return_value=automl_pipeline_instance_mock)
        monkeypatch.setattr("fermihamml.experiments.single_experiment.AutoMLPipeline", automl_pipeline_cls_mock)
        return automl_pipeline_instance_mock

    @pytest.fixture(autouse=True)
    def mock_output_folder(self, monkeypatch):
        output_folder_cls_mock = create_autospec(OutputFolder)
        monkeypatch.setattr("fermihamml.experiments.single_experiment.OutputFolder", output_folder_cls_mock)
        return output_folder_cls_mock

    def test_run_single_experiment_automl(self, instance_with_automl):
        result = instance_with_automl.run()
        assert result == 0, "SingleExperiment with automl did not run successfully."

    def test_run_single_experiment_lightning(self, instance_with_lightning):
        result = instance_with_lightning.run()
        assert result == 0, "SingleExperiment with LightningPipeline did not run successfully."

    def test_init_with_nones(self, config_with_automl):
        instance = SingleExperiment(
            config=config_with_automl,
            root_output_folder=None,
        )
        assert isinstance(instance._root_output_folder, Path)
        assert isinstance(instance.output_folder, OutputFolder)

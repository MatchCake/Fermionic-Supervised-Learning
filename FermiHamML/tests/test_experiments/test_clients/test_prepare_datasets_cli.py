import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from fermihamml.datasets import dataset_name_to_type_map
from fermihamml.experiments.clients.prepare_datasets_cli import prepare_datasets_cli
from fermihamml.experiments.configs.datasets_config import DatasetsConfig
from fermihamml.experiments.configs.experiment_config import ExperimentConfig
from fermihamml.experiments.configs.model_config import ModelConfig
from fermihamml.experiments.configs.opt_config import OptConfig


class TestLaunchJobsCli:
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        self.runner = CliRunner()
        yield

    @pytest.fixture
    def config_instance(self):
        return ExperimentConfig(
            experiment_name="test_experiment",
            datasets=DatasetsConfig(
                names=["test_dataset"],
                num_split_id=1,
                batch_sizes=[32],
                seed=0,
            ),
            models=[
                ModelConfig(
                    name="NIFIsingCls",
                    extra_options={"test_option": 0},
                )
            ],
            opt_config=OptConfig(
                automl_iterations=1,
                inner_max_epochs=1,
                max_epochs=1,
                automl_overwrite_fit=True,
            ),
        )

    @pytest.fixture(autouse=True)
    def config_yaml(self, config_instance):
        config_path = Path(".tmp/launch_jobs_cli") / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_instance.to_yaml_file(config_path)
        yield config_path
        shutil.rmtree(config_path.parent, ignore_errors=True)

    @pytest.fixture(autouse=True)
    def mock_get_dataset(self, monkeypatch):
        get_dataset_cls_by_name_mock = MagicMock()
        datasets_mocks = {name: MagicMock() for name in dataset_name_to_type_map}
        get_dataset_cls_by_name_mock.side_effect = lambda name: datasets_mocks[name]
        get_dataset_cls_by_name_mock.__datasets_mocks = datasets_mocks  # Store the mocks for later assertions
        monkeypatch.setattr(
            "fermihamml.experiments.clients.prepare_datasets_cli.get_dataset_cls_by_name", get_dataset_cls_by_name_mock
        )
        return get_dataset_cls_by_name_mock

    def test_cli_run(self, mock_get_dataset):
        result = self.runner.invoke(prepare_datasets_cli, [])
        assert result.exit_code == 0, f"CLI did not exit cleanly: {result.output}"
        mock_get_dataset.assert_called(), "Expected get_dataset_cls_by_name to be called, but it was not."
        datasets_called = [call.args[0] for call in mock_get_dataset.call_args_list]
        expected_datasets = [name for name in dataset_name_to_type_map]
        assert set(datasets_called) == set(
            expected_datasets
        ), f"Expected datasets {expected_datasets} to be prepared, but got {datasets_called}"
        for dataset_name, dataset_mock in mock_get_dataset.__datasets_mocks.items():
            assert (
                dataset_mock.return_value.prepare_data.called
            ), f"Expected prepare_data to be called for dataset {dataset_name}, but it was not."

import shutil
from pathlib import Path

import pytest

from fermihamml.result_analysis.experiment_results import ExperimentResults
from fermihamml.utils.output_folder import ExperimentState, OutputFolder


class TestExperimentResults:
    @pytest.fixture(autouse=True)
    def experiment_folder(self):
        experiment_folder = Path(".tmp/experiment_results")
        output_folders = [OutputFolder(experiment_folder / f"run_{i}") for i in range(10)]
        for i, output_folder in enumerate(output_folders):
            output_folder.update_data(
                {
                    ExperimentResults.DATASET_COL: f"dataset_{i % 3}",
                    ExperimentResults.MODEL_NAME_COL: f"model_{i % 2}",
                    ExperimentResults.PARAMS_COL: i * 1000,
                    ExperimentResults.TRAIN_ACC_COL: 0.8 + i * 0.01,
                    ExperimentResults.VAL_ACC_COL: 0.75 + i * 0.01,
                    ExperimentResults.TEST_ACC_COL: 0.78 + i * 0.01,
                }
            )
            output_folder.state = ExperimentState.FINISHED
        yield experiment_folder
        shutil.rmtree(experiment_folder, ignore_errors=True)

    def test_accuracies_df(self, experiment_folder):
        results = ExperimentResults(experiment_folder)
        acc_df = results.accuracies_df
        assert not acc_df.empty, "Accuracies DataFrame should not be empty."
        assert acc_df["Count"].sum() == 10, "Total count of runs should be 10."

    def test_jobs_status_df(self, experiment_folder):
        results = ExperimentResults(experiment_folder)
        status_df = results.jobs_status_df
        assert not status_df.empty, "Jobs status DataFrame should not be empty."
        assert status_df["FINISHED"].sum() == 10, "All jobs should be marked as FINISHED."

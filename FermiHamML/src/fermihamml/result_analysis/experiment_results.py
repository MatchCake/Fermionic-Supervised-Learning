import json
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from fermihamml.utils.output_folder import OutputFolder, ExperimentState


class ExperimentResults:
    TRAIN_ACC_COL = "train_MulticlassAccuracy"
    VAL_ACC_COL = "val_MulticlassAccuracy"
    TEST_ACC_COL = "test_MulticlassAccuracy"
    MODEL_NAME_COL = "model.name"
    PARAMS_COL = "num_params"
    DATASET_COL = "dataset_config.dataset_name"
    OUTPUT_FOLDER_COL = "_output_folder"

    @staticmethod
    def format_mean_std(mean: float, std: float, sig_std: int = 1) -> str:
        if pd.isna(mean) or pd.isna(std):  # pragma: no cover
            return "-"

        if std == 0:  # pragma: no cover
            return f"{mean} ± 0"

        # number of decimals needed for std
        decimals = -int(np.floor(np.log10(abs(std)))) + (sig_std - 1)
        decimals = max(decimals, 0)

        mean_rounded = round(mean, decimals)
        std_rounded = round(std, decimals)
        return f"{mean_rounded:.{decimals}f} ± {std_rounded:.{decimals}f}"

    def __init__(self, experiment_folder: Union[str, Path]):
        self.experiment_folder = Path(experiment_folder)
        self.metadata_df, self.data_df = OutputFolder.root_folder_to_dataframe(self.experiment_folder)
        self.output_folders = OutputFolder.gather_output_folders(self.experiment_folder)

    @cached_property
    def accuracies_df(self):
        r"""
        Return a table like the table 3 of https://arxiv.org/pdf/2504.10542. The table has the following columns:
            - Dataset: the name of the dataset
            - Model: the name of the model
            - Params: the number of parameters of the model (or "-" if not available)
            - Train: the mean and std of the training accuracy across different runs, formatted as "mean ± std"
            - Val: the mean and std of the validation accuracy across different runs, formatted as "mean ± std"
            - Test: the mean and std of the test accuracy across different runs, formatted as "mean ± std"
            - Count: the number of runs for this dataset and model
        """
        if self.PARAMS_COL not in self.data_df.columns:  # pragma: no cover
            self.data_df[self.PARAMS_COL] = None
        self.data_df[self.PARAMS_COL] = self.data_df[self.PARAMS_COL].apply(lambda x: "-" if x is None else x)
        acc_df = self.data_df[
            [
                self.DATASET_COL,
                self.MODEL_NAME_COL,
                self.PARAMS_COL,
                self.TRAIN_ACC_COL,
                self.VAL_ACC_COL,
                self.TEST_ACC_COL,
            ]
        ].copy()
        acc_df = acc_df.dropna(subset=[self.TRAIN_ACC_COL, self.VAL_ACC_COL, self.TEST_ACC_COL], how="any")
        acc_df[self.TRAIN_ACC_COL] = acc_df[self.TRAIN_ACC_COL] * 100
        acc_df[self.VAL_ACC_COL] = acc_df[self.VAL_ACC_COL] * 100
        acc_df[self.TEST_ACC_COL] = acc_df[self.TEST_ACC_COL] * 100

        train_acc_mean_col = "train_acc_mean"
        train_acc_std_col = "train_acc_std"
        val_acc_mean_col = "val_acc_mean"
        val_acc_std_col = "val_acc_std"
        test_acc_mean_col = "test_acc_mean"
        test_acc_std_col = "test_acc_std"
        n_run_col = "Count"
        aggregated_df = (
            acc_df.groupby([self.DATASET_COL, self.MODEL_NAME_COL, self.PARAMS_COL])
            .agg(
                **{
                    train_acc_mean_col: (self.TRAIN_ACC_COL, "mean"),
                    train_acc_std_col: (self.TRAIN_ACC_COL, "std"),
                    val_acc_mean_col: (self.VAL_ACC_COL, "mean"),
                    val_acc_std_col: (self.VAL_ACC_COL, "std"),
                    test_acc_mean_col: (self.TEST_ACC_COL, "mean"),
                    test_acc_std_col: (self.TEST_ACC_COL, "std"),
                    n_run_col: (self.TRAIN_ACC_COL, "count"),
                }
            )
            .reset_index()
        )
        aggregated_df[train_acc_mean_col] = aggregated_df.apply(
            lambda row: self.format_mean_std(row[train_acc_mean_col], row[train_acc_std_col]), axis=1
        )
        aggregated_df[val_acc_mean_col] = aggregated_df.apply(
            lambda row: self.format_mean_std(row[val_acc_mean_col], row[val_acc_std_col]), axis=1
        )
        aggregated_df[test_acc_mean_col] = aggregated_df.apply(
            lambda row: self.format_mean_std(row[test_acc_mean_col], row[test_acc_std_col]), axis=1
        )
        result_df = aggregated_df[
            [
                self.DATASET_COL,
                self.MODEL_NAME_COL,
                self.PARAMS_COL,
                train_acc_mean_col,
                val_acc_mean_col,
                test_acc_mean_col,
                n_run_col,
            ]
        ].copy()
        result_df.rename(
            columns={
                self.DATASET_COL: "Dataset",
                self.MODEL_NAME_COL: "Model",
                self.PARAMS_COL: "# Params",
                train_acc_mean_col: "Train [%]",
                val_acc_mean_col: "Val [%]",
                test_acc_mean_col: "Test [%]",
                n_run_col: "Count",
            },
            inplace=True,
        )
        return result_df

    @cached_property
    def jobs_status_df(self):
        data_df = self.data_df[
            [
                self.DATASET_COL,
                self.MODEL_NAME_COL,
                self.OUTPUT_FOLDER_COL,
            ]
        ].copy()
        data_df[self.OUTPUT_FOLDER_COL] = data_df[self.OUTPUT_FOLDER_COL].apply(lambda x: Path(x).name)
        status_col = "status"
        status_df = pd.DataFrame(
            [
                {
                    self.OUTPUT_FOLDER_COL: of.path.name,
                    status_col: of.state.value,
                }
                for of in self.output_folders
            ]
        )
        status_extended_df = data_df.merge(status_df, on=self.OUTPUT_FOLDER_COL, how="left")
        status_extended_df = (
            status_extended_df.groupby([self.DATASET_COL, self.MODEL_NAME_COL, status_col])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        status_extended_df.rename(
            columns={
                self.DATASET_COL: "Dataset",
                self.MODEL_NAME_COL: "Model",
            },
            inplace=True,
        )
        return status_extended_df

    @cached_property
    def errors_json_str(self):
        errors = defaultdict(list)
        for output_folder in self.output_folders:
            if output_folder.state == ExperimentState.FAILED:
                errors[output_folder.state_file.read()].append(output_folder.path.name)
        return json.dumps(errors, indent=2, sort_keys=True, default=str)

    @cached_property
    def errors_df(self):
        errors = defaultdict(list)
        for output_folder in self.output_folders:
            if output_folder.state == ExperimentState.FAILED:
                errors[output_folder.state_file.read()].append(output_folder.path.name)
        errors_df = pd.DataFrame(
            [
                {
                    "error": error,
                    "count": len(paths),
                    "paths": paths,
                }
                for error, paths in errors.items()
            ]
        )
        return errors_df

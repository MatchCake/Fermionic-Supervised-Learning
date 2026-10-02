import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from fermihamml.utils.output_folder import (
    ExperimentState,
    ExperimentStateFile,
    OutputFolder,
)


class TestExperimentStateFile:
    @pytest.fixture(autouse=True, scope="class")
    def state_file(self):
        return ExperimentStateFile(".tmp/output_folder/test_state_file")

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown(self, state_file):
        os.makedirs(".tmp/output_folder/test_state_file", exist_ok=True)
        yield
        shutil.rmtree(".tmp/output_folder/test_state_file", ignore_errors=True)

    def test_append(self, state_file):
        state_file.append("Test content")
        content = state_file.read()
        assert content == "Test content", "Content was not appended correctly."

    def test_state_setter(self, state_file):
        state_file.state = ExperimentState.FINISHED
        assert state_file.state == ExperimentState.FINISHED, "State was not set correctly."
        assert state_file.path.exists(), "State file was not created correctly."


class TestOutputFolder:
    @pytest.fixture
    def output_folder(self):
        return OutputFolder(
            ".tmp/output_folder/test_output_folder",
        )

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, output_folder):
        os.makedirs(".tmp/output_folder/test_output_folder", exist_ok=True)
        yield
        shutil.rmtree(".tmp/output_folder", ignore_errors=True)

    def test_update_metadata(self, output_folder):
        output_folder.update_metadata({"key": "value"})
        assert output_folder.metadata_file["key"] == "value", "Metadata was not updated correctly."

    def test_gather_files(self, output_folder):
        files = output_folder.gather_files()
        assert isinstance(files, list), "gather_files should return a list."
        assert len(files) == 3, "gather_files should return 3 files in the output folder."
        assert os.path.basename(output_folder.metadata_file.path) in [
            os.path.basename(f) for f in files
        ], "Metadata file should be included in the gathered files."

        with open(os.path.join(output_folder.path, "test_file.txt"), "w+") as f:
            f.write("Test content")

        files = output_folder.gather_files("*.txt")
        assert len(files) == 1, "gather_files should return one file matching the pattern."
        assert os.path.basename(files[0]) == "test_file.txt", "gather_files did not return the expected file."

    def test_root_folder_to_dataframe(self, output_folder):
        meta_df, df = OutputFolder.root_folder_to_dataframe(".tmp/output_folder")
        assert df is not None, "root_folder_to_dataframe should return a DataFrame."
        assert isinstance(df, pd.DataFrame), "root_folder_to_dataframe should return a DataFrame."
        assert meta_df is not None, "root_folder_to_dataframe should return a metadata DataFrame."
        assert isinstance(meta_df, pd.DataFrame), "root_folder_to_dataframe should return a metadata DataFrame."

    def test_fspath(self, output_folder):
        path_str = os.fspath(output_folder.path)
        assert isinstance(path_str, str), "fspath should return a string."
        assert path_str == os.fspath(output_folder), "fspath should return the same string as the output folder's path."

    def test_state_setter(self, output_folder):
        output_folder.state = ExperimentState.FINISHED
        assert output_folder.state == ExperimentState.FINISHED, "State was not set correctly."
        assert output_folder.state_file.path.exists(), "State file was not created correctly."

    def test_data_file_property(self, output_folder):
        data_file = output_folder.data_file
        assert Path(data_file.path).exists(), "Data file was not created correctly."
        assert Path(data_file.path) == output_folder.path / "run_output.out.json", "Data file path is incorrect."

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from fermihamml.datasets.imdb_dataset import IMDbDataset


class TestIMDbDataset:
    MOCK_LEN = 10

    @pytest.fixture(scope="class")
    def data_dir(self):
        path = Path(".tmp") / "data_dir" / "imdb"
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def imdb_mock(self, monkeypatch):
        imdb_cls_mock = MagicMock()
        monkeypatch.setattr("fermihamml.datasets.imdb_dataset.load_dataset", imdb_cls_mock)
        dataset_dict_mock = MagicMock()
        imdb_cls_mock.return_value = dataset_dict_mock
        dataset_mock = MagicMock()
        dataset_dict_mock.__getitem__.return_value = dataset_mock
        dataset_mock.__getitem__.return_value = {"text": "This is a test.", "label": 1}
        dataset_mock.__len__.return_value = self.MOCK_LEN

        word_embedder_cls_mock = MagicMock()
        word_embedder_mock = MagicMock()
        word_embedder_mock.embedding_dim = 50
        word_embedder_mock.return_value = torch.randn(5, 50)
        word_embedder_cls_mock.return_value = word_embedder_mock
        monkeypatch.setattr("fermihamml.datasets.nlp_base_dataset.WordEmbedder", word_embedder_cls_mock)

        return dataset_mock

    def test_init(self, imdb_mock, data_dir):
        dataset = IMDbDataset(data_dir=data_dir, train=True)
        assert dataset.train

    def test_getitem(self, imdb_mock, data_dir):
        dataset = IMDbDataset(data_dir=data_dir, train=True)
        datum = dataset[0]
        assert isinstance(datum, tuple)
        assert isinstance(datum[0], torch.Tensor)
        assert isinstance(datum[1], torch.Tensor)
        assert datum[1].dtype == torch.long
        imdb_mock.__getitem__.assert_called_with(0)

    def test_len(self, imdb_mock, data_dir):
        dataset = IMDbDataset(data_dir=data_dir, train=True)
        assert len(dataset) == self.MOCK_LEN

    def test_output_shape(self, imdb_mock, data_dir):
        dataset = IMDbDataset(data_dir=data_dir, train=True)
        assert dataset.get_output_shape() == (2,)

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from fermihamml.datasets.sst2_dataset import SST2Dataset


class TestSST2Dataset:
    MOCK_LEN = 10

    @pytest.fixture(scope="class")
    def data_dir(self):
        path = Path(".tmp") / "data_dir" / "sst2"
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def sst2_mock(self, monkeypatch):
        sst2_cls_mock = MagicMock()
        monkeypatch.setattr("fermihamml.datasets.sst2_dataset.load_dataset", sst2_cls_mock)
        dataset_dict_mock = MagicMock()
        sst2_cls_mock.return_value = dataset_dict_mock
        dataset_mock = MagicMock()
        dataset_dict_mock.__getitem__.return_value = dataset_mock
        dataset_mock.__getitem__.return_value = {"sentence": "This is a test.", "label": 1}
        dataset_mock.__len__.return_value = self.MOCK_LEN

        word_embedder_cls_mock = MagicMock()
        word_embedder_mock = MagicMock()
        word_embedder_mock.embedding_dim = 50
        word_embedder_mock.return_value = torch.randn(5, 50)
        word_embedder_cls_mock.return_value = word_embedder_mock
        monkeypatch.setattr("fermihamml.datasets.nlp_base_dataset.WordEmbedder", word_embedder_cls_mock)

        return dataset_mock

    def test_init(self, sst2_mock, data_dir):
        dataset = SST2Dataset(data_dir=data_dir, train=True)
        assert dataset.train

    def test_getitem(self, sst2_mock, data_dir):
        dataset = SST2Dataset(data_dir=data_dir, train=True)
        datum = dataset[0]
        assert isinstance(datum, tuple)
        assert isinstance(datum[0], torch.Tensor)
        assert isinstance(datum[1], torch.Tensor)
        assert datum[1].dtype == torch.long
        sst2_mock.__getitem__.assert_called_with(0)

    def test_len(self, sst2_mock, data_dir):
        dataset = SST2Dataset(data_dir=data_dir, train=True)
        assert len(dataset) == self.MOCK_LEN * 2  # train + validation

    def test_output_shape(self, sst2_mock, data_dir):
        dataset = SST2Dataset(data_dir=data_dir, train=True)
        assert dataset.get_output_shape() == (2,)

    def test_to_long_tensor(self):
        y = [0, 1, 2]
        tensor = SST2Dataset.to_long_tensor(y)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long
        assert tensor.tolist() == y

    def test_collate_fn(self, sst2_mock, data_dir):
        dataset = SST2Dataset(data_dir=data_dir, train=True)
        batch = [dataset[0], dataset[1]]
        padded_sentences, labels = SST2Dataset.collate_fn(batch)
        assert isinstance(padded_sentences, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert padded_sentences.shape[0] == 2
        assert labels.shape[0] == 2

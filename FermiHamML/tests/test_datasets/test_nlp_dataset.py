import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from fermihamml.datasets.nlp_base_dataset import NlpBaseDataset


class TestNlpBaseDataset:
    MOCK_LEN = 10

    @pytest.fixture(scope="class")
    def data_dir(self):
        path = Path(".tmp") / "data_dir" / "nlp_dataset"
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def nlp_mock(self, monkeypatch):
        word_embedder_cls_mock = MagicMock()
        word_embedder_mock = MagicMock()
        word_embedder_mock.embedding_dim = 50
        word_embedder_mock.return_value = torch.randn(5, 50)
        word_embedder_cls_mock.return_value = word_embedder_mock
        monkeypatch.setattr("fermihamml.datasets.nlp_base_dataset.WordEmbedder", word_embedder_cls_mock)
        return word_embedder_mock

    def test_init(self, nlp_mock, data_dir):
        dataset = NlpBaseDataset(data_dir=data_dir, train=True)
        assert dataset.train

    def test_to_long_tensor(self):
        y = [0, 1, 2]
        tensor = NlpBaseDataset.to_long_tensor(y)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long
        assert tensor.tolist() == y

    def test_collate_fn(self, nlp_mock, data_dir):
        dataset = NlpBaseDataset(data_dir=data_dir, train=True)
        batch = [[torch.randn(5, 50), torch.tensor(1)], [torch.randn(7, 50), torch.tensor(0)]]
        padded_sentences, labels = dataset.collate_fn(batch)
        assert isinstance(padded_sentences, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert padded_sentences.shape == (2, 7, 50)  # batch_size, max_seq_len, embedding_dim
        assert labels.shape == (2,)  # batch_size

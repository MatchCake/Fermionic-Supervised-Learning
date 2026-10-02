import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from fermihamml.datasets.fashion_dataset import FashionDataset


class TestFashionDataset:
    MOCK_LEN = 10

    @pytest.fixture(scope="class")
    def data_dir(self):
        path = Path(".tmp") / "data_dir" / "cifar2"
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def fashion_mock(self, monkeypatch):
        fashion_cls_mock = MagicMock()
        monkeypatch.setattr("fermihamml.datasets.fashion_dataset.FashionMNIST", fashion_cls_mock)
        mock = MagicMock()
        fashion_cls_mock.return_value = mock
        mock.__getitem__.return_value = (torch.zeros(28, 28), torch.zeros(1).long())
        mock.__len__.return_value = self.MOCK_LEN
        return mock

    def test_init(self, fashion_mock, data_dir):
        dataset = FashionDataset(data_dir=data_dir, train=True)
        assert dataset._data == fashion_mock

    def test_getitem(self, fashion_mock, data_dir):
        dataset = FashionDataset(data_dir=data_dir, train=True)
        datum = dataset[0]
        assert isinstance(datum, tuple)
        assert isinstance(datum[0], torch.Tensor)
        assert isinstance(datum[1], torch.Tensor)
        assert datum[1].dtype == torch.long
        fashion_mock.__getitem__.assert_called_once_with(0)

    def test_len(self, fashion_mock, data_dir):
        dataset = FashionDataset(data_dir=data_dir, train=True)
        assert len(dataset) == self.MOCK_LEN

    def test_output_shape(self, fashion_mock, data_dir):
        dataset = FashionDataset(data_dir=data_dir, train=True)
        assert dataset.get_output_shape() == (10,)

    def test_to_long_tensor(self):
        y = [0, 1, 2]
        tensor = FashionDataset.to_long_tensor(y)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long
        assert tensor.tolist() == y

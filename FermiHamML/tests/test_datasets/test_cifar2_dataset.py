import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from fermihamml.datasets.cifar2_dataset import Cifar2Dataset


class TestCifar2Dataset:
    MOCK_LEN = 10

    @pytest.fixture(scope="class")
    def data_dir(self):
        path = Path(".tmp") / "data_dir" / "cifar2"
        yield path
        shutil.rmtree(path, ignore_errors=True)

    @pytest.fixture
    def cifar10_mock(self, monkeypatch):
        cifar10_cls_mock = MagicMock()
        monkeypatch.setattr("fermihamml.datasets.cifar2_dataset.CIFAR10", cifar10_cls_mock)
        mock = MagicMock()
        cifar10_cls_mock.return_value = mock
        mock.__getitem__.return_value = (torch.zeros(28, 28), torch.zeros(1).long())
        mock.__len__.return_value = self.MOCK_LEN
        return mock

    def test_init(self, cifar10_mock, data_dir):
        dataset = Cifar2Dataset(data_dir=data_dir, train=True)
        assert dataset._data == cifar10_mock

    def test_getitem(self, cifar10_mock, data_dir):
        dataset = Cifar2Dataset(data_dir=data_dir, train=True)
        datum = dataset[0]
        assert isinstance(datum, tuple)
        assert isinstance(datum[0], torch.Tensor)
        assert isinstance(datum[1], torch.Tensor)
        assert datum[1].dtype == torch.long
        cifar10_mock.__getitem__.assert_called_once_with(0)

    def test_len(self, cifar10_mock, data_dir):
        dataset = Cifar2Dataset(data_dir=data_dir, train=True)
        assert len(dataset) == self.MOCK_LEN

    def test_output_shape(self, cifar10_mock, data_dir):
        dataset = Cifar2Dataset(data_dir=data_dir, train=True)
        assert dataset.get_output_shape() == (2,)

    def test_to_long_tensor(self):
        y = [0, 1, 2]
        tensor = Cifar2Dataset.to_long_tensor(y)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long
        assert tensor.tolist() == y

    def test_binarize_label(self):
        for label in [0, 1, 8, 9]:
            assert Cifar2Dataset.binarize_label(label) == 0
        for label in [2, 3, 4, 5, 6, 7]:
            assert Cifar2Dataset.binarize_label(label) == 1
        with pytest.raises(ValueError):
            Cifar2Dataset.binarize_label(10)

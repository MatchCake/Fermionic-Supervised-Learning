from functools import partial

import pytest

from fermihamml.datasets.collated_datamodule import CollatedDataModule


class TestCollatedDataModule:
    @pytest.fixture
    def datamodule_instance(self, monkeypatch):
        collate_fn_called = {"train": False, "val": False, "test": False}

        def collate_fn(batch, split):
            collate_fn_called[split] = True
            return list(zip(*batch))

        datamodule = CollatedDataModule.from_dataset_name("digits2d", 0, num_workers=0)
        datamodule.prepare_data()
        return datamodule, collate_fn, collate_fn_called

    def test_collate_fn_called(self, datamodule_instance):
        datamodule, collate_fn, collate_fn_called = datamodule_instance
        # call the dataloaders to trigger the collate_fn
        datamodule._given_train_dataset.collate_fn = partial(collate_fn, split="train")
        for _ in datamodule.train_dataloader():
            break
        datamodule._given_train_dataset.collate_fn = partial(collate_fn, split="val")
        for _ in datamodule.val_dataloader():
            break
        datamodule._test_dataset.collate_fn = partial(collate_fn, split="test")
        for _ in datamodule.test_dataloader():
            break

        assert collate_fn_called["train"]
        assert collate_fn_called["val"]
        assert collate_fn_called["test"]

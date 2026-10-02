from matchcake_opt import DataModule
from torch.utils.data import DataLoader


class CollatedDataModule(DataModule):
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            persistent_workers=self._num_workers > 0,
            pin_memory=True,
            collate_fn=getattr(self._given_train_dataset, "collate_fn", None),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            persistent_workers=self._num_workers > 0,
            pin_memory=True,
            collate_fn=getattr(self._given_train_dataset, "collate_fn", None),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            persistent_workers=self._num_workers > 0,
            pin_memory=True,
            collate_fn=getattr(self._test_dataset, "collate_fn", None),
        )

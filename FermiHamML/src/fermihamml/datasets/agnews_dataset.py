from pathlib import Path
from typing import Union

from datasets import load_dataset
from torch.utils.data import Dataset

from .nlp_base_dataset import NlpBaseDataset


class AGNewsDataset(NlpBaseDataset):
    """
    AG News dataset for text classification.

    Link: https://huggingface.co/datasets/fancyzhx/ag_news
    """

    DATASET_NAME = "AGNews"

    def __init__(self, data_dir: Union[str, Path] = Path("./data/") / DATASET_NAME, train: bool = True, **kwargs):
        super().__init__(data_dir, train, **kwargs)
        datasets = load_dataset("fancyzhx/ag_news", cache_dir=str(self.data_dir.resolve()))
        if self.train:
            self._data: Dataset = datasets["train"]
        else:
            self._data = datasets["test"]

    def __getitem__(self, item):
        sentence = self._data[item]["text"]
        label = self._data[item]["label"]
        target = self.to_long_tensor(label)
        embedded_sentence = self.preprocess_sentence(sentence)
        return embedded_sentence, target

    def __len__(self):
        return len(self._data)

    def get_output_shape(self) -> tuple:
        return (4,)

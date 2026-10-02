from pathlib import Path
from typing import Union

from datasets import load_dataset
from torch.utils.data import Dataset

from .nlp_base_dataset import NlpBaseDataset


class IMDbDataset(NlpBaseDataset):
    """
    IMDB movie review dataset for sentiment analysis.
     - Each review is labeled as either positive (1) or negative (0).
     - The dataset is split into training and test sets.
     - The training and the test sets both consist of 25,000 reviews each.

    Link: https://huggingface.co/datasets/stanfordnlp/imdb
    """

    DATASET_NAME = "IMDb"

    def __init__(self, data_dir: Union[str, Path] = Path("./data/") / DATASET_NAME, train: bool = True, **kwargs):
        super().__init__(data_dir, train, **kwargs)
        datasets = load_dataset("stanfordnlp/imdb", cache_dir=str(self.data_dir.resolve()))
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
        return (2,)

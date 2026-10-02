from pathlib import Path
from typing import Sequence, Union

import torch
from datasets import load_dataset
from matchcake_opt.datasets import BaseDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset

from ..modules.word_embedder import WordEmbedder
from .nlp_base_dataset import NlpBaseDataset


class SST2Dataset(NlpBaseDataset):
    """
    The Stanford Sentiment Treebank (SST-2) dataset is a widely used benchmark for sentiment analysis tasks.
    It consists of sentences labeled with binary sentiment labels (positive or negative).
    The dataset is derived from movie reviews and is part of the GLUE benchmark suite.

    Important note about the test set: The test set for SST-2 is not publicly available, so we will use the validation
    set for testing and combine the training and validation sets for training. This is only for debugging
    purposes and means that the metrics computed on the test set will means nothing. What should be done is to
    submit the predictions of the test set to the GLUE leaderboard to get the actual test metrics.

    TODO: Add a way to compute the actual test metrics by submitting to the GLUE leaderboard.
    """

    DATASET_NAME = "SST2"

    def __init__(self, data_dir: Union[str, Path] = Path("./data/") / DATASET_NAME, train: bool = True, **kwargs):
        super().__init__(data_dir, train, **kwargs)
        datasets = load_dataset("glue", "sst2", cache_dir=str(self.data_dir.resolve()))
        if self.train:
            self._data: Dataset = ConcatDataset([datasets["train"], datasets["validation"]])
        else:
            self._data = datasets["validation"]

    def __getitem__(self, item):
        sentence = self._data[item]["sentence"]
        label = self._data[item]["label"]
        target = self.to_long_tensor(label)
        embedded_sentence = self.preprocess_sentence(sentence)
        return embedded_sentence, target

    def __len__(self):
        return len(self._data)

    def get_output_shape(self) -> tuple:
        return (2,)

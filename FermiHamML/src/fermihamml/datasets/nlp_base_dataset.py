from pathlib import Path
from typing import Sequence, Union

import torch
from datasets import load_dataset
from matchcake_opt.datasets import BaseDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset

from ..modules.word_embedder import WordEmbedder


class NlpBaseDataset(BaseDataset):
    """ """

    DATASET_NAME = "_NlpBase"

    @staticmethod
    def to_long_tensor(y: Sequence[int]) -> torch.Tensor:
        return torch.tensor(y, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        sentences, labels = zip(*batch)
        padded_sentences = pad_sequence(sentences, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
        return padded_sentences, labels

    def __init__(self, data_dir: Union[str, Path] = Path("./data/") / DATASET_NAME, train: bool = True, **kwargs):
        super().__init__(data_dir, train, **kwargs)
        self.word_embedder = WordEmbedder(data_dir=data_dir)

    def preprocess_sentence(self, sentence: Union[str, Sequence[str]]) -> torch.Tensor:
        if isinstance(sentence, str):
            sentence = sentence.split()
        str_sentence = " ".join(sentence)
        embedded_sentence = self.word_embedder(str_sentence)
        return embedded_sentence

    def get_input_shape(self) -> tuple:
        return (self.word_embedder.embedding_dim,)

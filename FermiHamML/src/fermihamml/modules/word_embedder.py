import gzip
import os
from pathlib import Path
from typing import Union

import nltk
import requests
import torch
from gensim.models import KeyedVectors
from nltk import RegexpTokenizer
from tqdm import tqdm


class WordEmbedder(torch.nn.Module):
    EMBEDDING_URL = (
        "https://github.com/mmihaltz/word2vec-GoogleNews-vectors/raw/master/GoogleNews-vectors-negative300.bin.gz"
    )

    def __init__(
        self,
        *,
        freeze: bool = True,
        data_dir: Union[Path, str] = Path("./data/word_embeddings"),
        remove_gz: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self._freeze = freeze
        self.data_dir = Path(data_dir)
        self.remove_gz = remove_gz
        self.verbose = verbose
        if not self.bin_path.exists():
            self._download_embeddings()
            self._unzip_embeddings()
        nltk.download("punkt", download_dir=self.data_dir / "nltk_data")
        self.tokenizer = RegexpTokenizer(r"\w+")

        self.word2vec = KeyedVectors.load_word2vec_format(str(self.bin_path.resolve()), binary=True, limit=None)
        self.embedding = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(self.word2vec.vectors), freeze=self._freeze
        )

    def forward(self, text: str) -> torch.Tensor:
        tokens = self.tokenize(text)
        indices = [self.word2vec.key_to_index[token] for token in tokens if token in self.word2vec.key_to_index]
        if not indices:
            return torch.zeros(self.embedding.embedding_dim).reshape(1, -1)
        indices_tensor = torch.LongTensor(indices)
        torch_embedding = [self.embedding(idx) for idx in indices_tensor]
        torch_embedding = torch.nn.utils.rnn.pad_sequence(torch_embedding, batch_first=True)  # type: ignore
        return torch_embedding  # type: ignore

    def tokenize(self, text: str):
        return self.tokenizer.tokenize(text.lower())

    def _download_embeddings(self):
        self.gz_path.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.gz_path):
            with requests.get(self.EMBEDDING_URL, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length", 0))
                with (
                    open(self.gz_path, "wb") as f,
                    tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading embeddings to {self.gz_path}",
                        disable=not self.verbose,
                    ) as pbar,
                ):
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        return self

    def _unzip_embeddings(self):
        assert self.gz_path.exists(), f"Compressed file not found at {self.gz_path}. Please download it first."
        gz_size = os.path.getsize(self.gz_path)

        with (
            gzip.open(self.gz_path, "rb") as f_in,
            open(self.bin_path, "wb") as f_out,
            tqdm(
                total=gz_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"Extracting embeddings to {self.bin_path}",
                disable=not self.verbose,
            ) as pbar,
        ):
            while True:
                chunk = f_in.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)
                pbar.update(len(chunk))

        if self.remove_gz and self.gz_path.exists():
            os.remove(self.gz_path)
        return self

    @property
    def gz_path(self):
        return self.data_dir / "GoogleNews-vectors-negative300.bin.gz"

    @property
    def bin_path(self):
        return self.data_dir / "GoogleNews-vectors-negative300.bin"

    @property
    def embedding_dim(self):
        return self.embedding.embedding_dim

    @property
    def max_length(self):
        return len(self.word2vec.key_to_index)

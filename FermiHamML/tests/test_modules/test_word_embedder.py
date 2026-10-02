import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from fermihamml.modules.word_embedder import WordEmbedder


class TestWordEmbedder:
    @pytest.fixture(autouse=True, scope="class")
    def data_dir(self):
        folder = Path(".tmp/word_embeddings")
        yield folder
        shutil.rmtree(folder, ignore_errors=True)
        return

    @pytest.fixture
    def embedder_instance(self, monkeypatch, data_dir):
        request_mock = MagicMock()

        def iter_content(chunk_size=1024 * 1024):
            for _ in range(5):  # Simulate 5 chunks of data
                yield b"x" * chunk_size

        request_mock.iter_content = iter_content
        monkeypatch.setattr("fermihamml.modules.word_embedder.requests.get", request_mock)

        nltk_mock = MagicMock()
        monkeypatch.setattr("fermihamml.modules.word_embedder.nltk.download", nltk_mock)

        keyedvectors_mock = MagicMock()
        monkeypatch.setattr("fermihamml.modules.word_embedder.KeyedVectors.load_word2vec_format", keyedvectors_mock)
        word2vec_mock = MagicMock()
        word2vec_mock.vectors = [[0.1] * 300] * 100  # Simulate 100 words with 300-dimensional embeddings
        word2vec_mock.key_to_index = {f"word{i}": i for i in range(100)}
        keyedvectors_mock.return_value = word2vec_mock

        from_pretrained_mock = MagicMock()
        monkeypatch.setattr("torch.nn.Embedding.from_pretrained", from_pretrained_mock)
        embedding_mock = MagicMock()
        from_pretrained_mock.return_value = embedding_mock
        from_pretrained_mock.return_value.embedding_dim = 300
        embedding_mock.return_value = torch.FloatTensor([[0.1] * 300])  # Simulate the embedding output

        instance = WordEmbedder(freeze=True, verbose=False, data_dir=data_dir)
        instance.__request_mock = request_mock
        instance.__nltk_mock = nltk_mock
        instance.__keyedvectors_mock = keyedvectors_mock
        instance.__from_pretrained_mock = from_pretrained_mock
        return instance

    def test_init(self, embedder_instance, data_dir):
        embedder_instance.__nltk_mock.assert_called_with("punkt", download_dir=data_dir / "nltk_data")
        embedder_instance.__request_mock.assert_called_with(embedder_instance.EMBEDDING_URL, stream=True)
        embedder_instance.__from_pretrained_mock.assert_called()

    def test_forward(self, embedder_instance):
        text = "Hello world! word0 word1"
        embedding = embedder_instance(text)
        assert embedding.shape[-1] == embedder_instance.embedding.embedding_dim

    def test_tokenize(self, embedder_instance):
        text = "Hello world! word0 word1 word100"
        tokens = embedder_instance.tokenize(text)
        assert tokens == ["hello", "world", "word0", "word1", "word100"]

    def test_embedding_dim_property(self, embedder_instance):
        assert embedder_instance.embedding_dim == 300

    def test_maxlength_property(self, embedder_instance):
        assert embedder_instance.max_length == 100

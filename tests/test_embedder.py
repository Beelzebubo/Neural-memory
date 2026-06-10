import importlib.util
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# Module-level check: is sentence-transformers actually installed in this environment?
_HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec("sentence_transformers") is not None


@pytest.fixture
def mock_sentence_transformers():
    mock_module = MagicMock()
    mock_instance = MagicMock()
    mock_instance.get_embedding_dimension.return_value = 384
    mock_instance.encode.return_value = np.array([0.1] * 384, dtype=np.float32)
    mock_module.SentenceTransformer.return_value = mock_instance
    sys.modules["sentence_transformers"] = mock_module
    yield mock_instance
    del sys.modules["sentence_transformers"]


@pytest.fixture
def embedder(mock_sentence_transformers):
    from src.embedder import TextEmbedder
    return TextEmbedder(model_name="test-model")


class TestTextEmbedder:
    def test_embed_empty_string_returns_zero_vector(self, embedder, mock_sentence_transformers):
        result = embedder.embed("")
        expected = [0.0] * 384
        assert result == expected

    def test_embed_whitespace_string_returns_zero_vector(self, embedder, mock_sentence_transformers):
        result = embedder.embed("   ")
        expected = [0.0] * 384
        assert result == expected

    def test_embed_valid_text_returns_correct_dimension(self, embedder, mock_sentence_transformers):
        result = embedder.embed("hello world")
        assert len(result) == 384

    def test_embed_normalizes_embeddings(self):
        """Embedding should be L2-normalized (unit length)."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_instance = MagicMock()
            # encode(text) with a string returns 1D for SentenceTransformer
            mock_instance.encode.return_value = np.array([3.0, 4.0], dtype=np.float32)
            mock_instance.get_embedding_dimension.return_value = 2
            mock_st.return_value = mock_instance

            from src.embedder import TextEmbedder
            emb = TextEmbedder(model_name="test")
            result = emb.embed("test")

        # The mock doesn't normalize — the test verifies normalize_embeddings=True is forwarded
        call_kwargs = mock_instance.encode.call_args[1]
        assert call_kwargs.get("normalize_embeddings") is True

    def test_embed_batch_empty_list_returns_empty(self, embedder, mock_sentence_transformers):
        result = embedder.embed_batch([])
        assert result == []

    def test_embed_batch_valid_texts(self, embedder, mock_sentence_transformers):
        texts = ["hello", "world"]

        def encode_side_effect(t, **kwargs):
            arr = np.zeros((len(t), 384), dtype=np.float32)
            for i in range(len(t)):
                arr[i] = np.array([0.1 * (i + 1)] * 384, dtype=np.float32)
            return arr

        mock_sentence_transformers.encode.side_effect = encode_side_effect

        result = embedder.embed_batch(texts)
        assert len(result) == 2
        assert len(result[0]) == 384

    def test_embed_batch_mixed_empty_and_valid(self, embedder, mock_sentence_transformers):
        texts = ["hello", "", "world"]

        def encode_side_effect(t, **kwargs):
            arr = np.zeros((len(t), 384), dtype=np.float32)
            for i in range(len(t)):
                if t[i].strip():
                    arr[i] = np.array([0.1 * (i + 1)] * 384, dtype=np.float32)
            return arr

        mock_sentence_transformers.encode.side_effect = encode_side_effect

        result = embedder.embed_batch(texts)
        assert len(result) == 3
        assert result[1] == [0.0] * 384

    def test_len_returns_embedding_dimension(self, embedder):
        assert len(embedder) == 384

    def test_init_raises_without_sentence_transformers(self):
        """When sentence-transformers is not installed, should raise ImportError."""
        if _HAS_SENTENCE_TRANSFORMERS:
            pytest.skip("sentence-transformers is installed — test only valid without it")
        saved = sys.modules.pop("sentence_transformers", None)
        try:
            if "src.embedder" in sys.modules:
                del sys.modules["src.embedder"]
            from src.embedder import TextEmbedder
            with pytest.raises(ImportError, match="sentence-transformers is required"):
                TextEmbedder(model_name="test-model")
        finally:
            if saved is not None:
                sys.modules["sentence_transformers"] = saved

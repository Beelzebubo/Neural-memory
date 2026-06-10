from typing import List, Optional

import numpy as np


class TextEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install it with: pip install sentence-transformers"
            )
        self._dim = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * self._dim
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        cleaned = [t if t and t.strip() else "" for t in texts]
        vecs = self._model.encode(cleaned, normalize_embeddings=True)
        results = []
        for i, v in enumerate(vecs):
            if not cleaned[i]:
                results.append([0.0] * self._dim)
            else:
                results.append(v.tolist())
        return results

    def __len__(self) -> int:
        return self._dim

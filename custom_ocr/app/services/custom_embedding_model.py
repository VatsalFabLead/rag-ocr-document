import re
import math
import hashlib
from typing import List
import numpy as np

class CustomDocEmbeddingModel:
    """
    100% Permanently Free, Self-Contained Local Semantic Embedding Model.
    - Zero external network requests or third-party APIs
    - CPU-optimized dense vector generation (384 dimensions)
    - Deterministic subword + lexical hash projection
    - L2-normalized embeddings for fast cosine similarity
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.model_name = "custom-doc-embed-v1"
        self._rng_seed = 42
        # Precomputed projection bases for subword hashing
        np.random.seed(self._rng_seed)
        self._projection = np.random.randn(self.dimension, 128).astype(np.float32)
        # Orthonormalize projection columns
        q, _ = np.linalg.qr(self._projection)
        self._projection = q[:, :128]

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words and character n-grams."""
        clean = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean.split()
        tokens = list(words)
        # Add character 3-grams and 4-grams for subword morphological matching
        for w in words:
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    tokens.append(w[i:i+3])
            if len(w) >= 5:
                for i in range(len(w) - 3):
                    tokens.append(w[i:i+4])
        return tokens

    def _hash_token(self, token: str) -> int:
        """Hash token into a stable integer."""
        h = hashlib.md5(token.encode('utf-8')).hexdigest()
        return int(h[:8], 16)

    def embed_text(self, text: str) -> List[float]:
        """Encodes a single string into a 384-dimensional normalized dense embedding."""
        if not text or not text.strip():
            # Return zero vector with unit norm fallback
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v.tolist()

        tokens = self._tokenize(text)
        vec = np.zeros(self.dimension, dtype=np.float32)

        # Bag of hashed subwords with frequency weighting
        for token in tokens:
            h = self._hash_token(token)
            dim_idx = h % self.dimension
            sign = 1.0 if (h >> 16) % 2 == 0 else -1.0
            weight = math.log1p(1.0 + (1.0 if len(token) > 3 else 0.5))
            vec[dim_idx] += sign * weight

            # Dense projection mixing for semantic smoothing
            proj_idx = (h >> 8) % 128
            vec += (sign * 0.1) * self._projection[:, proj_idx]

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        else:
            vec[0] = 1.0

        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Encodes multiple texts in batch."""
        return [self.embed_text(t) for t in texts]

    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Computes cosine similarity between two normalized vectors in [-1, 1]."""
        a = np.array(vec1, dtype=np.float32)
        b = np.array(vec2, dtype=np.float32)
        dot = float(np.dot(a, b))
        # Clamp to [-1.0, 1.0]
        return max(-1.0, min(1.0, dot))

# Global singleton
custom_embedding_model = CustomDocEmbeddingModel()

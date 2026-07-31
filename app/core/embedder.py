"""
Local Text Embedder — No external API required.

Uses scikit-learn's HashingVectorizer to produce fixed-size, consistent
text embeddings that support cosine similarity comparisons.

HashingVectorizer advantages:
  - No fitting step needed (vocabulary is hashed, not learned)
  - Deterministic: same text always gives same vector
  - Fast, lightweight, zero network calls
  - Works offline and on Vercel serverless
"""

import numpy as np
from typing import List, Union
from sklearn.feature_extraction.text import HashingVectorizer


class ResumeEmbedder:
    """
    Converts resume and job description text into fixed-dimensional
    normalized vectors using TF-style hashing. Cosine similarity between
    two such vectors measures semantic overlap.
    """

    def __init__(self, n_features: int = 2 ** 14):
        """
        Args:
            n_features: Size of the hashed feature space (default 16 384).
                        Larger values reduce hash collisions.
        """
        self.n_features = n_features
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            norm="l2",            # already unit-length → cosine sim = dot product
            alternate_sign=False, # keep all positive for cleaner similarity
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams capture phrases like "machine learning"
            analyzer="word",
            lowercase=True,
        )

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generates an L2-normalized embedding vector for the input text.

        Args:
            text: A single string or a list of strings.
        Returns:
            np.ndarray of shape (n_features,) for a single string,
            or (n_texts, n_features) for a list.
        """
        if isinstance(text, str):
            mat = self._vectorizer.transform([text])
            return np.asarray(mat.todense(), dtype=np.float32).squeeze(0)
        else:
            mat = self._vectorizer.transform(text)
            return np.asarray(mat.todense(), dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        return self.n_features


# Global singleton — zero initialization cost, no network calls
embedder = ResumeEmbedder()

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union

class ResumeEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the sentence embedder with a SentenceTransformer model.
        """
        self.model_name = model_name
        # The model will be loaded on demand (lazy loading) to make start-up fast
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generates embedding vector(s) for the input text(s).
        Returns a numpy array of shape (embedding_dim,) or (num_texts, embedding_dim).
        """
        if isinstance(text, str):
            # Encode a single string
            embeddings = self.model.encode([text], convert_to_numpy=True)
            return embeddings[0]
        else:
            # Encode a list of strings
            return self.model.encode(text, convert_to_numpy=True)

    @property
    def embedding_dim(self) -> int:
        """
        Returns the dimensionality of the generated embeddings.
        """
        # For 'all-MiniLM-L6-v2', this is 384.
        return self.model.get_sentence_embedding_dimension()

# Instantiate a global embedder for use across the application
embedder = ResumeEmbedder()

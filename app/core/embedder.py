import numpy as np
import os
from typing import List, Union
from google import genai

class ResumeEmbedder:
    def __init__(self, model_name: str = "text-embedding-004"):
        """
        Initializes the sentence embedder with Gemini API.
        """
        self.model_name = model_name
        self._client = None

    @property
    def client(self):
        if self._client is None:
            # Requires GEMINI_API_KEY environment variable to be set
            self._client = genai.Client()
        return self._client

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generates embedding vector(s) for the input text(s).
        Returns a numpy array of shape (embedding_dim,) or (num_texts, embedding_dim).
        """
        if isinstance(text, str):
            # Encode a single string
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            return np.array(response.embeddings[0].values, dtype=np.float32)
        else:
            # Encode a list of strings
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            embeddings = [emb.values for emb in response.embeddings]
            return np.array(embeddings, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        """
        Returns the dimensionality of the generated embeddings.
        """
        return 768

# Instantiate a global embedder for use across the application
embedder = ResumeEmbedder()

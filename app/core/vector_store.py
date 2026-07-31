import os
import pickle
import faiss
import numpy as np
from typing import Dict, List, Tuple, Any

class ResumeVectorStore:
    def __init__(self, embedding_dim: int = 768, persist_dir: str = "data/vector_store"):
        """
        Initializes FAISS IndexFlatIP (Inner Product) for cosine similarity.
        Normalizes embeddings prior to insertion and search.
        """
        self.embedding_dim = embedding_dim
        # Vercel filesystem is read-only except for /tmp
        if os.environ.get("VERCEL"):
            self.persist_dir = "/tmp/vector_store"
        else:
            self.persist_dir = persist_dir
            
        self.index_file = os.path.join(self.persist_dir, "faiss.index")
        self.meta_file = os.path.join(self.persist_dir, "metadata.pkl")

        # Create persistence directory if it doesn't exist
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = faiss.IndexFlatIP(self.embedding_dim)
        # Dictionary mapping FAISS index position to resume details (filename, raw_text, id)
        self.metadata_map: Dict[int, Dict[str, Any]] = {}
        # Counter for FAISS IDs (since FAISS index is 0-indexed contiguous positions)
        self.current_id = 0

        # Attempt to load existing index
        self.load()

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalizes a single vector or a batch of vectors for cosine similarity.
        """
        if vector.ndim == 1:
            norm = np.linalg.norm(vector)
            if norm == 0:
                return vector
            return vector / norm
        else:
            norms = np.linalg.norm(vector, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vector / norms

    def add_resume(self, resume_id: str, filename: str, text: str, embedding: np.ndarray):
        """
        Adds a resume to the vector store.
        """
        normalized_emb = self._normalize(embedding)
        
        # Reshape to 2D array if 1D for FAISS
        if normalized_emb.ndim == 1:
            normalized_emb = np.expand_dims(normalized_emb, axis=0)

        # Add to FAISS index
        self.index.add(normalized_emb.astype('float32'))

        # Store metadata mapping
        self.metadata_map[self.current_id] = {
            "id": resume_id,
            "filename": filename,
            "text": text
        }
        self.current_id += 1
        self.save()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches the FAISS index for the top K closest resumes to the query embedding.
        Returns a list of tuples containing (resume_metadata, score).
        """
        if self.index.ntotal == 0:
            return []

        normalized_query = self._normalize(query_embedding)
        if normalized_query.ndim == 1:
            normalized_query = np.expand_dims(normalized_query, axis=0)

        # Ensure top_k is not larger than total index size
        k = min(top_k, self.index.ntotal)
        
        # FAISS search
        # D: distances (inner product, i.e. cosine similarity since vectors are normalized)
        # I: indices
        D, I = self.index.search(normalized_query.astype('float32'), k)

        results = []
        for rank in range(k):
            idx = I[0][rank]
            score = float(D[0][rank])
            if idx in self.metadata_map:
                # Convert inner product score (cosine similarity range [-1, 1])
                # to a percentage range [0, 100]% or keep it as 0 to 1 decimal.
                # Let's return raw cosine similarity score.
                results.append((self.metadata_map[idx], score))
                
        return results

    def save(self):
        """
        Persists the FAISS index and the metadata mapping.
        """
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.meta_file, 'wb') as f:
                pickle.dump((self.metadata_map, self.current_id), f)
        except Exception as e:
            print(f"Failed to persist vector store: {e}")

    def load(self):
        """
        Loads the FAISS index and metadata mapping from disk if they exist.
        """
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.meta_file, 'rb') as f:
                    self.metadata_map, self.current_id = pickle.load(f)
            except Exception as e:
                print(f"Failed to load vector store: {e}. Reinitializing empty store.")
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.metadata_map = {}
                self.current_id = 0

    def clear(self):
        """
        Clears the store completely.
        """
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata_map = {}
        self.current_id = 0
        self.save()

    def get_all_resumes(self) -> List[Dict[str, Any]]:
        """
        Returns all stored resumes and metadata.
        """
        return list(self.metadata_map.values())

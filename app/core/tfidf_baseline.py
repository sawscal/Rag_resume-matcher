from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple, Any

class TFIDFBaselineMatcher:
    def __init__(self):
        pass

    def match(self, query: str, resumes: List[Dict[str, Any]], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Matches a job description query against stored resumes using TF-IDF and Cosine Similarity.
        """
        if not resumes:
            return []

        # Extract text from resumes
        resume_texts = [r["text"] for r in resumes]
        
        # Initialize TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            # Fit and transform resume texts
            tfidf_matrix = vectorizer.fit_transform(resume_texts)
            
            # Transform query
            query_vector = vectorizer.transform([query])
            
            # Compute cosine similarities
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            
            # Sort matches by similarity score descending
            indices = np.argsort(similarities)[::-1]
            
            results = []
            k = min(top_k, len(resumes))
            for rank in range(k):
                idx = indices[rank]
                score = float(similarities[idx])
                results.append((resumes[idx], score))
                
            return results
        except Exception as e:
            # If TF-IDF fitting fails (e.g. empty vocabulary), return matches with 0 score
            return [(resumes[i], 0.0) for i in range(min(top_k, len(resumes)))]

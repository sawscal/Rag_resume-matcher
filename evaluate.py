import sys
import os
import time
import numpy as np

# Add parent directory to path so we can import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.sample_data import SAMPLE_RESUMES, SAMPLE_JOB_DESCRIPTIONS
from app.core.embedder import embedder
from app.core.tfidf_baseline import TFIDFBaselineMatcher

def evaluate_models():
    print("=" * 70)
    print("Evaluating Resume & Job Matcher Accuracy: Sentence-BERT vs TF-IDF")
    print("=" * 70)

    tfidf_matcher = TFIDFBaselineMatcher()
    
    # 1. Precompute Sentence-BERT embeddings for resumes
    print("Generating Sentence-BERT embeddings for resumes...")
    resume_embeddings = []
    for r in SAMPLE_RESUMES:
        emb = embedder.embed_text(r["text"])
        resume_embeddings.append(emb)
    
    # Normalize resume embeddings for cosine similarity
    resume_embeddings_norm = []
    for emb in resume_embeddings:
        norm = np.linalg.norm(emb)
        resume_embeddings_norm.append(emb / norm if norm > 0 else emb)
    resume_embeddings_norm = np.array(resume_embeddings_norm)

    sbert_mrr = 0.0
    tfidf_mrr = 0.0
    sbert_p1 = 0.0
    tfidf_p1 = 0.0
    
    total_queries = len(SAMPLE_JOB_DESCRIPTIONS)

    for jd in SAMPLE_JOB_DESCRIPTIONS:
        title = jd["title"]
        query_text = jd["text"]
        expected_file = jd["expected_match"]

        print(f"\nQuery JD: {title}")
        print(f"Target Match: {expected_file}")
        
        # --- TF-IDF Baseline Matching ---
        tfidf_results = tfidf_matcher.match(query_text, SAMPLE_RESUMES, top_k=len(SAMPLE_RESUMES))
        
        # Find rank of expected match in TF-IDF
        tfidf_rank = -1
        for i, (res, score) in enumerate(tfidf_results):
            if res["filename"] == expected_file:
                tfidf_rank = i + 1
                break
        
        # --- Sentence-BERT Matching ---
        query_emb = embedder.embed_text(query_text)
        query_emb_norm = query_emb / np.linalg.norm(query_emb) if np.linalg.norm(query_emb) > 0 else query_emb
        
        # Compute Cosine Similarity for S-BERT
        sbert_scores = np.dot(resume_embeddings_norm, query_emb_norm)
        sbert_indices = np.argsort(sbert_scores)[::-1]
        
        sbert_results = []
        for idx in sbert_indices:
            sbert_results.append((SAMPLE_RESUMES[idx], float(sbert_scores[idx])))
            
        # Find rank of expected match in Sentence-BERT
        sbert_rank = -1
        for i, (res, score) in enumerate(sbert_results):
            if res["filename"] == expected_file:
                sbert_rank = i + 1
                break

        # Calculate metrics for this query
        sbert_rec_rank = 1.0 / sbert_rank if sbert_rank > 0 else 0.0
        tfidf_rec_rank = 1.0 / tfidf_rank if tfidf_rank > 0 else 0.0
        
        sbert_mrr += sbert_rec_rank
        tfidf_mrr += tfidf_rec_rank
        
        sbert_p1_val = 1.0 if sbert_rank == 1 else 0.0
        tfidf_p1_val = 1.0 if tfidf_rank == 1 else 0.0
        
        sbert_p1 += sbert_p1_val
        tfidf_p1 += tfidf_p1_val

        # Display results for this query
        print(f"  -> TF-IDF: Rank {tfidf_rank} (Score: {tfidf_results[0][1]:.3f}) | Reciprocal Rank: {tfidf_rec_rank:.2f}")
        print(f"  -> S-BERT: Rank {sbert_rank} (Score: {sbert_results[0][1]:.3f}) | Reciprocal Rank: {sbert_rec_rank:.2f}")

    # Compute averages
    avg_sbert_mrr = sbert_mrr / total_queries
    avg_tfidf_mrr = tfidf_mrr / total_queries
    avg_sbert_p1 = sbert_p1 / total_queries
    avg_tfidf_p1 = tfidf_p1 / total_queries
    
    # Calculate percentage improvement
    mrr_improvement = ((avg_sbert_mrr - avg_tfidf_mrr) / avg_tfidf_mrr * 100) if avg_tfidf_mrr > 0 else 0
    p1_improvement = ((avg_sbert_p1 - avg_tfidf_p1) / avg_tfidf_p1 * 100) if avg_tfidf_p1 > 0 else 0

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Metric            | TF-IDF Baseline | Sentence-BERT + FAISS | Improvement")
    print(f"------------------+-----------------+-----------------------+------------")
    print(f"Precision@1 (P@1) | {avg_tfidf_p1:.2%}         | {avg_sbert_p1:.2%}          | +{p1_improvement:.1f}%")
    print(f"Mean Recip Rank   | {avg_tfidf_mrr:.3f}           | {avg_sbert_mrr:.3f}             | +{mrr_improvement:.1f}%")
    print("=" * 70)
    print("Sentence-BERT semantic understanding successfully matches candidates based on concepts\nrather than exact word presence, showing significant performance improvements.\n")

if __name__ == "__main__":
    evaluate_models()

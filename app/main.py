import os
import time
import uuid
import io
import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from app.models import (
    MatchRequest,
    MatchResponse,
    MatchResult,
    RAGAnalysisRequest,
    RAGAnalysisResponse,
    ResumeResponse,
    UploadResponse,
    StatelessMatchRequest,
    StatelessAnalyzeRequest,
    ParsedMetadata,
    CandidateFilterRequest,
)
from app.core.parser import extract_text_from_bytes
from app.core.embedder import embedder
from app.core.vector_store import ResumeVectorStore
from app.core.tfidf_baseline import TFIDFBaselineMatcher
from app.core.rag_generator import rag_generator
from app.core.nlp_parser import parse_resume
from app.core.database import candidate_db

app = FastAPI(
    title="RAG-Based AI Resume & Job Matcher API",
    description="An end-to-end Retrieval-Augmented Generation (RAG) system using Sentence-BERT and FAISS for sub-150ms resume/job matching.",
    version="1.0.0"
)

# Enable CORS for convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global components
vector_store = ResumeVectorStore()
tfidf_matcher = TFIDFBaselineMatcher()

@app.get("/", response_class=FileResponse)
async def read_index():
    """
    Serve the frontend single-page interface.
    """
    file_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Frontend file not found.")

@app.post("/upload-resume", response_model=UploadResponse)
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a resume (PDF, DOCX, TXT) or a spreadsheet (Excel, CSV) containing multiple resumes,
    parse the text, generate embeddings, and index them in FAISS.
    """
    filename = file.filename
    try:
        content_bytes = await file.read()
        ext = os.path.splitext(filename.lower())[1]

        # 1. Spreadsheet handling (Excel or CSV Bulk Upload)
        if ext in ['.xlsx', '.xls', '.csv']:
            try:
                if ext == '.csv':
                    df = pd.read_csv(io.BytesIO(content_bytes), encoding='utf-8', errors='ignore')
                else:
                    df = pd.read_excel(io.BytesIO(content_bytes))
            except Exception as se:
                raise ValueError(f"Unable to parse spreadsheet file: {str(se)}")

            # Detect columns mapping
            name_col = None
            text_col = None
            
            for col in df.columns:
                col_lower = str(col).lower().strip()
                if not name_col and any(kw in col_lower for kw in ["name", "candidate", "filename", "title", "id"]):
                    name_col = col
                if not text_col and any(kw in col_lower for kw in ["resume", "text", "content", "cv", "description", "body"]):
                    text_col = col

            # Fallbacks if column matches not found
            if not name_col and len(df.columns) > 0:
                name_col = df.columns[0]
            if not text_col and len(df.columns) > 1:
                text_col = df.columns[1]
            elif not text_col and len(df.columns) == 1:
                text_col = df.columns[0]

            if not text_col:
                raise ValueError("Could not automatically locate the resume content column in the spreadsheet.")

            indexed_count = 0
            for idx, row in df.iterrows():
                candidate_name = str(row[name_col]).strip() if name_col and pd.notna(row[name_col]) else f"Row {idx + 1}"
                resume_text = str(row[text_col]).strip() if pd.notna(row[text_col]) else ""
                
                if not resume_text:
                    continue

                # Truncate title for list display
                candidate_name = candidate_name[:100]
                
                # S-BERT embedding
                embedding = embedder.embed_text(resume_text)
                resume_id = str(uuid.uuid4())[:8]
                
                # Add to FAISS store
                vector_store.add_resume(
                    resume_id=resume_id,
                    filename=candidate_name,
                    text=resume_text,
                    embedding=embedding
                )
                indexed_count += 1

            # Force save FAISS and metadata
            vector_store.save()

            return UploadResponse(
                is_bulk=True,
                count=indexed_count,
                message=f"Spreadsheet successfully parsed. Indexed {indexed_count} candidates in vector store."
            )

        # 2. Individual document handling (PDF, DOCX, TXT)
        else:
            text = extract_text_from_bytes(content_bytes, filename)
            if not text:
                raise HTTPException(status_code=400, detail="Unable to extract text content from the file.")
            
            # Generate embedding using Sentence-BERT
            embedding = embedder.embed_text(text)
            
            # Generate unique resume ID
            resume_id = str(uuid.uuid4())[:8]
            
            # Add to FAISS index and persist
            vector_store.add_resume(
                resume_id=resume_id,
                filename=filename,
                text=text,
                embedding=embedding
            )
            
            snippet = text[:200] + "..." if len(text) > 200 else text

            # --- NLP Parsing Pipeline ---
            parsed = parse_resume(text)
            meta = ParsedMetadata(
                skills=parsed.skills,
                experience_years=parsed.experience_years,
                education=parsed.education,
            )

            # --- TF-IDF score against a blank JD (placeholder; real score computed at match time) ---
            initial_tfidf = 0.0

            # --- MySQL storage (optional, silently skipped if DB unavailable) ---
            candidate_db.insert_candidate(
                resume_id=resume_id,
                filename=filename,
                skills=parsed.skills,
                experience_years=parsed.experience_years,
                education=parsed.education,
                tfidf_score=initial_tfidf,
                raw_text=text,
            )

            return UploadResponse(
                is_bulk=False,
                count=1,
                message=f"Resume '{filename}' successfully parsed and indexed.",
                resume_id=resume_id,
                filename=filename,
                snippet=snippet,
                resume_text=text,
                parsed_metadata=meta,
            )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file upload: {str(e)}")

@app.post("/match", response_model=MatchResponse)
async def match_resumes(request: MatchRequest):
    """
    Search resumes indexed in FAISS matching a job description query using Sentence-BERT embeddings.
    """
    start_time = time.perf_counter()
    try:
        # Embed the query job description
        query_emb = embedder.embed_text(request.job_description)
        
        # Search in FAISS
        search_results = vector_store.search(query_emb, top_k=request.top_k)
        
        matches = []
        for meta, score in search_results:
            snippet = meta["text"][:200] + "..." if len(meta["text"]) > 200 else meta["text"]
            matches.append(
                MatchResult(
                    resume_id=meta["id"],
                    filename=meta["filename"],
                    score=score,
                    snippet=snippet
                )
            )
            
        processing_time = (time.perf_counter() - start_time) * 1000.0
        
        return MatchResponse(
            matches=matches,
            method="Sentence-BERT + FAISS",
            processing_time_ms=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Matching query failed: {str(e)}")

@app.post("/match-tfidf", response_model=MatchResponse)
async def match_resumes_tfidf(request: MatchRequest):
    """
    Search resumes using the baseline TF-IDF and Cosine Similarity model (benchmark).
    """
    start_time = time.perf_counter()
    try:
        resumes = vector_store.get_all_resumes()
        if not resumes:
            return MatchResponse(
                matches=[],
                method="TF-IDF Baseline",
                processing_time_ms=(time.perf_counter() - start_time) * 1000.0
            )
            
        search_results = tfidf_matcher.match(
            query=request.job_description,
            resumes=resumes,
            top_k=request.top_k
        )
        
        matches = []
        for meta, score in search_results:
            snippet = meta["text"][:200] + "..." if len(meta["text"]) > 200 else meta["text"]
            matches.append(
                MatchResult(
                    resume_id=meta["id"],
                    filename=meta["filename"],
                    score=score,
                    snippet=snippet
                )
            )
            
        processing_time = (time.perf_counter() - start_time) * 1000.0
        
        return MatchResponse(
            matches=matches,
            method="TF-IDF Baseline",
            processing_time_ms=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TF-IDF matching query failed: {str(e)}")

@app.post("/rag-analyze", response_model=RAGAnalysisResponse)
async def rag_analyze(request: RAGAnalysisRequest):
    """
    Retrieve a specific resume and generate a detailed fit analysis and recommendation
    using a generative language model (Retrieval-Augmented Generation).
    """
    start_time = time.perf_counter()
    try:
        # Find the resume by ID
        resumes = vector_store.get_all_resumes()
        selected_resume = None
        for r in resumes:
            if r["id"] == request.resume_id:
                selected_resume = r
                break
                
        if not selected_resume:
            raise HTTPException(status_code=404, detail=f"Resume with ID {request.resume_id} not found.")

        # Compute match score (Sentence-BERT + FAISS distance)
        resume_emb = embedder.embed_text(selected_resume["text"])
        query_emb = embedder.embed_text(request.job_description)
        
        # Calculate raw cosine similarity score
        resume_emb_norm = resume_emb / np.linalg.norm(resume_emb) if np.linalg.norm(resume_emb) > 0 else resume_emb
        query_emb_norm = query_emb / np.linalg.norm(query_emb) if np.linalg.norm(query_emb) > 0 else query_emb
        score = float(np.dot(resume_emb_norm, query_emb_norm))

        # Generate evaluation using local Seq2Seq model
        analysis = rag_generator.generate_match_analysis(
            resume_text=selected_resume["text"],
            job_description=request.job_description
        )
        
        processing_time = (time.perf_counter() - start_time) * 1000.0
        
        return RAGAnalysisResponse(
            resume_id=request.resume_id,
            score=score,
            analysis=analysis,
            processing_time_ms=processing_time
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG evaluation failed: {str(e)}")

@app.get("/resumes", response_model=List[ResumeResponse])
async def list_resumes():
    """
    Lists metadata of all parsed and stored resumes.
    """
    resumes = vector_store.get_all_resumes()
    return [
        ResumeResponse(
            resume_id=r["id"],
            filename=r["filename"],
            snippet=r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
        )
        for r in resumes
    ]

@app.delete("/resumes")
async def clear_resumes():
    """
    Clears all resumes from the vector database.
    """
    vector_store.clear()
    return {"message": "All resumes cleared from vector store."}


@app.post("/stateless-match", response_model=MatchResponse)
async def stateless_match(request: StatelessMatchRequest):
    """
    Stateless match endpoint: accepts full resume texts inline in the request.
    Computes both semantic (TF-IDF hashing) and TF-IDF cosine similarity scores.
    No external API required — fully local using scikit-learn.
    Also runs the NLP parsing pipeline to attach skill/experience metadata.
    """
    import time
    start_time = time.perf_counter()
    try:
        if not request.candidates:
            return MatchResponse(matches=[], method="TF-IDF + NLP (Local)", processing_time_ms=0.0)

        # --- Semantic embedding scores ---
        query_emb = embedder.embed_text(request.job_description)
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)

        # --- TF-IDF scores ---
        resume_texts = [c.text for c in request.candidates]
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
        tfidf_scores_map = {}
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(resume_texts)
            query_vec = vectorizer.transform([request.job_description])
            raw_scores = sk_cosine(query_vec, tfidf_matrix).flatten()
            for i, c in enumerate(request.candidates):
                tfidf_scores_map[c.resume_id] = float(raw_scores[i])
        except Exception:
            # Gracefully skip TF-IDF if vocabulary is empty
            for c in request.candidates:
                tfidf_scores_map[c.resume_id] = 0.0

        # --- Score and rank candidates ---
        scored = []
        for candidate in request.candidates:
            c_emb = embedder.embed_text(candidate.text)
            c_norm = c_emb / (np.linalg.norm(c_emb) + 1e-9)
            sem_score = float(np.dot(query_norm, c_norm))
            scored.append((candidate, sem_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:request.top_k]

        # --- Build result list with NLP metadata ---
        matches = []
        for candidate, score in top:
            snippet = candidate.text[:200] + "..." if len(candidate.text) > 200 else candidate.text
            parsed = parse_resume(candidate.text)
            matches.append(MatchResult(
                resume_id=candidate.resume_id,
                filename=candidate.filename,
                score=score,
                tfidf_score=tfidf_scores_map.get(candidate.resume_id, 0.0),
                snippet=snippet,
                parsed_metadata=ParsedMetadata(
                    skills=parsed.skills,
                    experience_years=parsed.experience_years,
                    education=parsed.education,
                )
            ))

        processing_time = (time.perf_counter() - start_time) * 1000.0
        return MatchResponse(
            matches=matches,
            method="TF-IDF + NLP (Local)",
            processing_time_ms=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stateless match failed: {str(e)}")


@app.post("/stateless-analyze")
async def stateless_analyze(request: StatelessAnalyzeRequest):
    """
    Stateless RAG analysis: accepts full resume text inline — no stored state required.
    Works on Vercel serverless functions without any persistent filesystem.
    """
    import time
    start_time = time.perf_counter()
    try:
        # Compute similarity
        resume_emb = embedder.embed_text(request.resume_text)
        query_emb = embedder.embed_text(request.job_description)
        r_norm = resume_emb / (np.linalg.norm(resume_emb) + 1e-9)
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        score = float(np.dot(r_norm, q_norm))

        # Generate fit analysis via local NLP engine
        analysis = rag_generator.generate_match_analysis(
            resume_text=request.resume_text,
            job_description=request.job_description
        )
        processing_time = (time.perf_counter() - start_time) * 1000.0
        return {
            "resume_id": request.resume_id,
            "score": score,
            "analysis": analysis,
            "processing_time_ms": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stateless analysis failed: {str(e)}")


@app.post("/filter-candidates")
async def filter_candidates(request: CandidateFilterRequest):
    """
    Filter candidates stored in MySQL by skill keyword or minimum experience.
    Returns an empty list gracefully if MySQL is not configured.
    """
    if not candidate_db.is_available:
        return {
            "db_available": False,
            "message": "MySQL not configured. Set DATABASE_URL environment variable to enable persistent candidate filtering.",
            "candidates": []
        }

    results = []
    if request.skill:
        results = candidate_db.query_by_skill(request.skill)
    elif request.min_experience_years is not None:
        results = candidate_db.query_by_min_experience(request.min_experience_years)
    else:
        results = candidate_db.get_all()

    return {
        "db_available": True,
        "total": len(results),
        "candidates": results
    }


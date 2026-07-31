from pydantic import BaseModel, Field
from typing import List, Optional

class MatchRequest(BaseModel):
    job_description: str = Field(..., description="The job description text to match resumes against.")
    top_k: Optional[int] = Field(5, description="Number of top matches to retrieve.")

# Stateless models — resume text sent by client, no server-side storage needed
class StatelessCandidate(BaseModel):
    resume_id: str = Field(..., description="Client-generated candidate ID.")
    filename: str = Field(..., description="Original filename of the resume.")
    text: str = Field(..., description="Full extracted resume text.")

class StatelessMatchRequest(BaseModel):
    job_description: str = Field(..., description="The job description text to match against.")
    candidates: List[StatelessCandidate] = Field(..., description="Candidates with full resume text.")
    top_k: Optional[int] = Field(5, description="Number of top candidates to return.")

class StatelessAnalyzeRequest(BaseModel):
    job_description: str = Field(..., description="The job description text.")
    resume_id: str = Field(..., description="Candidate ID.")
    filename: str = Field(..., description="Resume filename.")
    resume_text: str = Field(..., description="Full text of the resume.")

class MatchResult(BaseModel):
    resume_id: str = Field(..., description="The unique ID of the resume.")
    filename: str = Field(..., description="The original filename of the resume.")
    score: float = Field(..., description="Similarity score (Cosine Similarity).")
    snippet: str = Field(..., description="Snippet of the resume text.")

class MatchResponse(BaseModel):
    matches: List[MatchResult] = Field(..., description="List of matched resumes sorted by relevance.")
    method: str = Field(..., description="The matching method used (e.g. Sentence-BERT + FAISS or TF-IDF Baseline).")
    processing_time_ms: float = Field(..., description="Time taken to process the query in milliseconds.")

class RAGAnalysisRequest(BaseModel):
    resume_id: str = Field(..., description="The unique ID of the resume to analyze.")
    job_description: str = Field(..., description="The job description text.")

class RAGAnalysisResponse(BaseModel):
    resume_id: str = Field(..., description="The analyzed resume ID.")
    score: float = Field(..., description="FAISS cosine similarity match score.")
    analysis: str = Field(..., description="Generative match analysis and recommendation.")
    processing_time_ms: float = Field(..., description="Time taken to run analysis in milliseconds.")

class ResumeResponse(BaseModel):
    resume_id: str
    filename: str
    snippet: str

class UploadResponse(BaseModel):
    is_bulk: bool = Field(False, description="True if the uploaded file was an Excel/CSV spreadsheet parsed in bulk.")
    count: int = Field(1, description="Number of candidates successfully indexed.")
    message: str = Field(..., description="General success notification message.")
    
    # Backward compatibility attributes for single uploads
    resume_id: Optional[str] = Field(None, description="The ID of the indexed resume (single upload).")
    filename: Optional[str] = Field(None, description="The filename of the indexed resume (single upload).")
    snippet: Optional[str] = Field(None, description="The text snippet of the indexed resume (single upload).")
    resume_text: Optional[str] = Field(None, description="Full extracted text of the resume (returned for client-side caching).")

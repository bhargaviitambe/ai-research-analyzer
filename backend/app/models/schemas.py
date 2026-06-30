from pydantic import BaseModel
from typing import List, Optional


# ---------- Upload ----------

class UploadResponse(BaseModel):
    paper_id: str
    filename: str
    message: str


# ---------- Summarization ----------

class SummaryResponse(BaseModel):
    abstract_summary: str
    methodology_summary: str
    results_summary: str
    conclusion_summary: str


# ---------- Research Gaps ----------

class ResearchGapsResponse(BaseModel):
    gaps: List[str]


# ---------- Future Work ----------

class FutureWorkResponse(BaseModel):
    suggestions: List[str]


# ---------- Q&A ----------

class QARequest(BaseModel):
    paper_id: str
    question: str


class QAResponse(BaseModel):
    question: str
    answer: str
    source_chunks: List[str]


# ---------- Citations ----------

class CitationResponse(BaseModel):
    authors: List[str]
    year: Optional[str]
    title: str
    apa: str
    mla: str
    ieee: str


# ---------- Similar Papers (your new feature) ----------

class SimilarPaper(BaseModel):
    title: str
    authors: List[str]
    year: Optional[str]
    link: str
    summary: Optional[str] = None


class SimilarPapersResponse(BaseModel):
    topic: str
    papers: List[SimilarPaper]
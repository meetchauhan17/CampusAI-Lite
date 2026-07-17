"""
api/main.py
===========
FastAPI backend exposing two endpoints that wrap the four agentic pipelines.

Start:
    uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import sys
import os
import time
from typing import List, Literal, Optional

# Ensure project root is in sys.path (relevant when running from api/ directly)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import Settings
from core.logger import logger
from core.llm_provider import LLMProvider

# ─────────────────────────────────────────────────────────────────────────────
# App & shared state
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CampusAI Lite API",
    description="FastAPI backend wrapping four agentic pipeline implementations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_settings = Settings()
_provider = LLMProvider(_settings)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

PipelineId = Literal["crewai", "langgraph", "autogen", "beeai"]


class AskRequest(BaseModel):
    question: str
    pipeline: PipelineId = "langgraph"


class ValidationResultResponse(BaseModel):
    pipeline: str
    final_answer: str
    category: str
    confidence: float
    is_grounded: bool
    is_accurate: bool
    issues: List[str]
    latency_ms: float
    error: Optional[str] = None


class CompareRequest(BaseModel):
    question: str


class CompareResponse(BaseModel):
    question: str
    results: List[ValidationResultResponse]


# ─────────────────────────────────────────────────────────────────────────────
# Internal runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(pipeline: str, question: str) -> ValidationResultResponse:
    t0 = time.perf_counter()
    try:
        if pipeline == "crewai":
            from agents.crewai_impl.crew import run_crewai_pipeline
            result = run_crewai_pipeline(question, _settings, _provider)
            category = _guess_category(result.final_answer)
            return ValidationResultResponse(
                pipeline=pipeline,
                final_answer=result.final_answer,
                category=category,
                confidence=result.confidence,
                is_grounded=result.is_grounded,
                is_accurate=result.is_accurate,
                issues=result.issues,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        elif pipeline == "langgraph":
            from agents.langgraph_impl.graph import run_langgraph_pipeline
            result = run_langgraph_pipeline(question, _settings, _provider)
            category = _guess_category(result.final_answer)
            return ValidationResultResponse(
                pipeline=pipeline,
                final_answer=result.final_answer,
                category=category,
                confidence=result.confidence,
                is_grounded=result.is_grounded,
                is_accurate=result.is_accurate,
                issues=result.issues,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        elif pipeline == "autogen":
            from agents.autogen_impl.group import run_autogen_pipeline
            result = run_autogen_pipeline(question, _settings, _provider)
            category = _guess_category(result.final_answer)
            return ValidationResultResponse(
                pipeline=pipeline,
                final_answer=result.final_answer,
                category=category,
                confidence=result.confidence,
                is_grounded=result.is_grounded,
                is_accurate=result.is_accurate,
                issues=result.issues,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        elif pipeline == "beeai":
            from agents.beeai_impl.poc import run_beeai_poc
            raw = run_beeai_poc(question, _settings, _provider)
            return ValidationResultResponse(
                pipeline=pipeline,
                final_answer=raw.get("answer", "No answer returned."),
                category=raw.get("category", "general"),
                confidence=0.70,
                is_grounded=True,
                is_accurate=True,
                issues=[],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        else:
            raise ValueError(f"Unknown pipeline: {pipeline}")

    except Exception as exc:
        logger.error("[API] Pipeline '{}' failed: {}", pipeline, exc)
        return ValidationResultResponse(
            pipeline=pipeline,
            final_answer="An error occurred while running the pipeline.",
            category="general",
            confidence=0.0,
            is_grounded=False,
            is_accurate=False,
            issues=[str(exc)],
            latency_ms=(time.perf_counter() - t0) * 1000,
            error=str(exc),
        )


def _guess_category(answer: str) -> str:
    kw_map = {
        "exam": "exams",
        "fee": "fees",
        "library": "library",
        "hostel": "hostel",
        "calendar": "academic-calendar",
    }
    lower = answer.lower()
    for kw, cat in kw_map.items():
        if kw in lower:
            return cat
    return "general"


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "CampusAI Lite API", "version": "1.0.0", "status": "ok"}


@app.post("/api/ask", response_model=ValidationResultResponse)
def ask(req: AskRequest):
    """Run a single pipeline and return its ValidationResult-shaped response."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    return _run_pipeline(req.pipeline, req.question.strip())


@app.post("/api/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    """Run all four pipelines and return their results with latency figures."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    q = req.question.strip()
    results = []
    for pipeline in ["crewai", "langgraph", "autogen", "beeai"]:
        results.append(_run_pipeline(pipeline, q))
    return CompareResponse(question=q, results=results)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

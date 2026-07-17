"""
agents/crewai_impl/crew.py

Wires the Planner, Information, and Validation agents + tasks into a
sequential crewai.Crew, and exposes:

    run_crewai_pipeline(user_question: str) -> ValidationResult

which the UI layer (Gradio) will call directly.

verbose is controlled by Settings.CREW_VERBOSE (default True for dev).
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from crewai import Crew, Process

from config.settings import Settings
from core.logger import logger
from core.llm_provider import LLMProvider
from core.schemas import ValidationResult
from agents.crewai_impl.agents import build_agents
from agents.crewai_impl.tasks import build_tasks


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Robustly extract the first JSON object from a string that may contain
    prose, markdown fences, or other surrounding text.
    """
    # Strip markdown code fences
    stripped = re.sub(r"```(?:json)?", "", text).strip().strip("`")

    # Try direct parse
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Fallback: find first {...} block
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from agent output:\n{text[:500]}")


def _safe_validation_result(raw_text: str, fallback_answer: str = "") -> ValidationResult:
    """Parse ValidationResult from agent output, with a safe fallback."""
    try:
        data = _extract_json(raw_text)
        return ValidationResult(
            is_grounded=bool(data.get("is_grounded", False)),
            is_accurate=bool(data.get("is_accurate", False)),
            final_answer=str(data.get("final_answer", fallback_answer)),
            confidence=float(data.get("confidence", 0.0)),
            issues=list(data.get("issues", [])),
        )
    except Exception as exc:
        logger.warning("Could not parse ValidationResult from crew output: {}. Using fallback.", exc)
        return ValidationResult(
            is_grounded=False,
            is_accurate=False,
            final_answer=fallback_answer or raw_text[:500],
            confidence=0.0,
            issues=[f"Failed to parse structured output: {exc}"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_crewai_pipeline(
    user_question: str,
    settings: Optional[Settings] = None,
    provider: Optional[LLMProvider] = None,
) -> ValidationResult:
    """
    Run the full three-agent sequential CrewAI pipeline and return a
    guaranteed-schema ValidationResult.

    Args:
        user_question: The student's raw query string.
        settings:      Project settings (loaded from env if not supplied).
        provider:      Shared LLMProvider instance (created if not supplied).

    Returns:
        ValidationResult — always, never raises to caller.
    """
    if settings is None:
        settings = Settings()

    if provider is None:
        provider = LLMProvider(settings)

    verbose: bool = getattr(settings, "CREW_VERBOSE", True)

    logger.info("[CrewAI] Starting pipeline for question: '{}'", user_question[:80])

    try:
        # Build agents & tasks
        agents = build_agents(provider=provider, verbose=verbose)
        plan_task, retrieve_task, validate_task = build_tasks(
            agents=agents,
            user_question=user_question,
        )

        # Assemble and run the Crew
        crew = Crew(
            agents=list(agents.values()),
            tasks=[plan_task, retrieve_task, validate_task],
            process=Process.sequential,
            verbose=verbose,
        )

        crew_result = crew.kickoff()

        # crew_result.raw is the final task's raw output string
        raw_output: str = (
            crew_result.raw
            if hasattr(crew_result, "raw")
            else str(crew_result)
        )

        logger.info("[CrewAI] Pipeline completed. Parsing ValidationResult.")
        result = _safe_validation_result(raw_output, fallback_answer="")
        logger.info(
            "[CrewAI] Result — is_accurate={}, confidence={:.2f}, issues={}",
            result.is_accurate,
            result.confidence,
            result.issues,
        )
        return result

    except Exception as exc:
        logger.error("[CrewAI] Pipeline failed: {}", exc)
        return ValidationResult(
            is_grounded=False,
            is_accurate=False,
            final_answer="Sorry, the system encountered an error. Please try again.",
            confidence=0.0,
            issues=[f"Pipeline exception: {type(exc).__name__}: {str(exc)[:300]}"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# __main__ verification block
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint

    SAMPLE_QUESTIONS = [
        "When is the Artificial Intelligence exam and what hall do I report to?",
        "How much is the B.Tech tuition fee and what happens if I pay late?",
        "How many books can a postgraduate student borrow from the library?",
    ]

    settings = Settings()
    provider = LLMProvider(settings)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n{'='*70}")
        print(f"[{i}/{len(SAMPLE_QUESTIONS)}] Question: {question}")
        print("="*70)

        result = run_crewai_pipeline(
            user_question=question,
            settings=settings,
            provider=provider,
        )

        print("\nValidationResult:")
        pprint.pprint(result.model_dump())
        print()

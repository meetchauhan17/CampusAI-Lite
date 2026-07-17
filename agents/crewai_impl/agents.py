"""
agents/crewai_impl/agents.py

Defines the three crewai.Agent instances used in the CrewAI pipeline:
  1. Planner Agent      — breaks the question into sub-tasks, detects category
  2. Information Agent  — uses UniversityInfoSearchTool to retrieve grounded facts
  3. Validation Agent   — delegates to the PydanticAI ValidationAgent for schema-safe output

All three agents use CampusAICrewLLM, a thin crewai.BaseLLM subclass that
routes every LiteLLM call through our LLMProvider (watsonx → groq → gemini).
This keeps a single code-path for fallover regardless of which framework is active.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from crewai import Agent
from crewai.llm import BaseLLM
from pydantic import model_validator

from config.settings import Settings
from core.logger import logger
from core.llm_provider import LLMProvider
from tools.university_search_tool import UniversityInfoSearchToolCrewAI


# ─────────────────────────────────────────────────────────────────────────────
# CrewAI-compatible LLM bridge: routes calls through LLMProvider
# ─────────────────────────────────────────────────────────────────────────────

class CampusAICrewLLM(BaseLLM):
    """
    A crewai.BaseLLM subclass that forwards every inference call to LLMProvider
    instead of going through LiteLLM / litellm directly.

    This satisfies the requirement that all three agents use the LangChain-adapter
    of LLMProvider (the same provider used by the other framework implementations),
    while remaining fully compatible with crewai's Agent machinery.
    """

    # Provider is stored as Any to pass Pydantic's arbitrary_types_allowed
    _provider: Any = None

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, provider: LLMProvider, **kwargs: Any) -> None:
        # crewai BaseLLM requires a model name string
        super().__init__(model="campus-ai-provider", **kwargs)
        object.__setattr__(self, "_provider", provider)

    # ── Required abstract implementation ────────────────────────────────────

    def call(
        self,
        messages: str | list[Any],
        tools: list[Any] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        """Synchronously invoke LLMProvider and return the text string."""
        system_prompt: Optional[str] = None
        user_prompt: str = ""

        if isinstance(messages, str):
            user_prompt = messages
        elif isinstance(messages, list):
            for m in messages:
                role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
                content = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
                if role == "system":
                    system_prompt = content
                else:
                    user_prompt += (f"\n{content}" if user_prompt else content)

        try:
            response = self._provider.generate(
                prompt=user_prompt,
                system=system_prompt,
            )
            logger.debug(
                "[CampusAICrewLLM] Provider used: {}, tokens: {}",
                response.provider_used,
                response.tokens_used,
            )
            return response.text
        except Exception as exc:
            logger.error("[CampusAICrewLLM] LLMProvider.generate() failed: {}", exc)
            raise

    # ── Optional helpers crewai may call ────────────────────────────────────

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8192


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────────────────

def build_agents(provider: LLMProvider, verbose: bool = True) -> dict[str, Agent]:
    """
    Builds and returns the three CrewAI Agent instances.

    Args:
        provider: Shared LLMProvider instance used by all agents.
        verbose:  If True, agents will log their reasoning steps.
    """
    llm = CampusAICrewLLM(provider=provider)
    search_tool = UniversityInfoSearchToolCrewAI()

    # ── 1. Planner Agent ────────────────────────────────────────────────────
    planner = Agent(
        role="University Query Planner",
        goal=(
            "Analyse the student's question and produce a concise execution plan. "
            "Determine the correct category (exams, fees, library, hostel, "
            "academic-calendar, or general) and list the sub-tasks needed to answer it. "
            "Output a JSON object with keys: sub_tasks (list[str]), category (str), "
            "requires_tool (bool)."
        ),
        backstory=(
            "You are an expert academic advisor AI that has been trained on the GTU "
            "campus knowledge base. You excel at decomposing student queries into clear, "
            "actionable retrieval sub-tasks so that downstream agents can answer accurately."
        ),
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
    )

    # ── 2. Information Agent ─────────────────────────────────────────────────
    information = Agent(
        role="University Information Retriever",
        goal=(
            "Use the university_info_search_tool to fetch relevant, grounded facts from "
            "university documents and draft a clear answer for the student. "
            "Output a JSON object with keys: raw_answer (str), sources (list[str]), "
            "category (str)."
        ),
        backstory=(
            "You are a university librarian AI with access to the official GTU document "
            "repository. You always cite the source files you used and never invent facts. "
            "You MUST ALWAYS call the search tool to fetch current facts for any student query. "
            "Do not skip calling the tool under any circumstances."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
    )

    # ── 3. Validation Agent ──────────────────────────────────────────────────
    validator = Agent(
        role="Answer Validator",
        goal=(
            "Validate that the information agent's answer is fully grounded in the "
            "retrieved source chunks. Correct small errors if possible. "
            "Output a JSON object with keys: is_grounded (bool), is_accurate (bool), "
            "final_answer (str), confidence (float 0-1), issues (list[str])."
        ),
        backstory=(
            "You are a meticulous fact-checker AI. You compare every claim in the drafted "
            "answer against the raw_chunks retrieved by the Information Agent and flag "
            "any hallucination or inaccuracy. You produce a structured ValidationResult "
            "that the UI layer can trust completely."
        ),
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
    )

    return {
        "planner": planner,
        "information": information,
        "validator": validator,
    }

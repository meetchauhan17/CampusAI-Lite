"""
Validation Agent — implemented with PydanticAI (satisfies the mandatory PydanticAI requirement).

The agent wraps the LLMProvider (via its FunctionModel adapter) and is bound to
`output_type=ValidationResult` so PydanticAI enforces the schema natively rather
than relying on manual JSON parsing.

Responsibilities:
  1. Given a user question + InformationAgentOutput + retrieved source chunks,
     verify that the raw_answer is actually grounded in those chunks.
  2. Correct small inaccuracies when possible; set is_accurate=False otherwise.
  3. Self-repair loop: if PydanticAI raises a ValidationError, retry once with
     an explicit schema-error prompt before giving up.
  4. Always return a structurally valid ValidationResult — never raises to caller.
"""
from __future__ import annotations

import asyncio
import json
from typing import List, Optional, Any

from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    SystemPromptPart,
    UserPromptPart,
)

from core.logger import logger
from core.schemas import InformationAgentOutput, ValidationResult

# ────────────────────────────────────────────────────────────────────────────
# PydanticAI FunctionModel bridge to LLMProvider
# ────────────────────────────────────────────────────────────────────────────

def _extract_text_from_messages(messages: list[ModelMessage]) -> tuple[str, Optional[str]]:
    """Walk a PydanticAI message list and return (user_prompt, system_prompt)."""
    system_prompt: Optional[str] = None
    user_parts: list[str] = []

    for msg in messages:
        for part in msg.parts:
            kind = getattr(part, "part_kind", "")
            if kind == "system-prompt":
                system_prompt = getattr(part, "content", "")
            elif kind == "user-prompt":
                content = getattr(part, "content", "")
                if isinstance(content, str):
                    user_parts.append(content)
                elif isinstance(content, (list, tuple)):
                    user_parts.append(
                        " ".join(
                            c if isinstance(c, str) else getattr(c, "text", "")
                            for c in content
                        )
                    )
            elif kind == "text":
                user_parts.append(getattr(part, "content", ""))

    return "\n".join(user_parts), system_prompt


def build_pydantic_ai_function_model(llm_provider: Any) -> FunctionModel:
    """
    Build a PydanticAI FunctionModel that routes requests through LLMProvider.
    The function is async and returns a ModelResponse with a single TextPart.
    """
    async def campus_llm_function(
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        user_prompt, system_prompt = _extract_text_from_messages(messages)
        response = await llm_provider.generate_async(
            prompt=user_prompt,
            system=system_prompt,
        )
        return ModelResponse(parts=[TextPart(content=response.text)])

    return FunctionModel(function=campus_llm_function, model_name="campus-llm-bridge")


# ────────────────────────────────────────────────────────────────────────────
# System prompt for the Validation Agent
# ────────────────────────────────────────────────────────────────────────────

VALIDATION_SYSTEM_PROMPT = """
You are a strict university information validation assistant.

Your job:
1. Read the user's original question, the raw answer provided by the Information Agent,
   and the retrieved document chunks that should support that answer.
2. Decide if the raw_answer is grounded in the retrieved chunks (no hallucinated facts).
3. If you spot a factual error, correct it in `final_answer`.
4. If the answer is unverifiable from the chunks, set is_grounded=False and is_accurate=False.
5. Always produce a confidence score between 0.0 and 1.0.
6. Always fill the `issues` list with specific problem descriptions (leave empty if there are none).

You MUST respond with a JSON object matching this exact schema and nothing else:
{
  "is_grounded": <bool>,
  "is_accurate": <bool>,
  "final_answer": "<string>",
  "confidence": <float between 0.0 and 1.0>,
  "issues": ["<issue description>", ...]
}

Do not include markdown code fences, prose, or any other text outside the JSON object.
""".strip()

SCHEMA_REPAIR_PROMPT = """
Your previous response did not match the required JSON schema. Here is the validation error:

{error}

Please fix your response and return ONLY a valid JSON object with these exact keys:
  is_grounded (bool), is_accurate (bool), final_answer (str), confidence (float 0.0-1.0), issues (list[str])

No markdown, no explanations — just the JSON object.
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Validation Agent class
# ────────────────────────────────────────────────────────────────────────────

class ValidationAgent:
    """
    PydanticAI-powered Validation Agent with structured-output guarantee.

    Uses `Agent(output_type=ValidationResult)` so the library enforces the schema.
    Falls back to a self-repair retry on ValidationError, and returns a safe
    ValidationResult(is_accurate=False) if both attempts fail.
    """

    def __init__(self, llm_provider: Any) -> None:
        self._llm_provider = llm_provider
        self._model = build_pydantic_ai_function_model(llm_provider)
        self._agent: Agent[None, ValidationResult] = Agent(
            model=self._model,
            output_type=ValidationResult,
            system_prompt=VALIDATION_SYSTEM_PROMPT,
        )
        logger.info("ValidationAgent initialized with PydanticAI (output_type=ValidationResult).")

    def _build_user_prompt(
        self,
        user_question: str,
        info_output: InformationAgentOutput,
        source_chunks: List[dict],
    ) -> str:
        chunks_text = "\n---\n".join(
            f"[Source: {c.get('source_file', 'unknown')} | Category: {c.get('category', 'n/a')}]\n{c.get('text', '')}"
            for c in source_chunks
        ) or "No source chunks were retrieved."

        return (
            f"Original user question:\n{user_question}\n\n"
            f"Raw answer from Information Agent:\n{info_output.raw_answer}\n\n"
            f"Cited sources: {', '.join(info_output.sources) or 'none'}\n\n"
            f"Retrieved document chunks:\n{chunks_text}"
        )

    async def validate_async(
        self,
        user_question: str,
        info_output: InformationAgentOutput,
        source_chunks: Optional[List[dict]] = None,
    ) -> ValidationResult:
        """
        Async validation. Attempts PydanticAI structured-output run, then retries
        once on schema failure before returning a safe fallback result.
        """
        user_prompt = self._build_user_prompt(
            user_question, info_output, source_chunks or []
        )

        # ── First attempt ────────────────────────────────────────────────────
        try:
            logger.info("ValidationAgent: first attempt for question='{}'", user_question[:80])
            result = await self._agent.run(user_prompt)
            logger.info(
                "ValidationAgent: success — is_accurate={}, confidence={:.2f}",
                result.output.is_accurate,
                result.output.confidence,
            )
            return result.output

        except (ValidationError, Exception) as first_exc:
            logger.warning(
                "ValidationAgent: first attempt failed ({}). Retrying with repair prompt.",
                type(first_exc).__name__,
            )
            error_detail = str(first_exc)

        # ── Repair retry ─────────────────────────────────────────────────────
        repair_prompt = (
            f"{user_prompt}\n\n"
            + SCHEMA_REPAIR_PROMPT.format(error=error_detail)
        )
        try:
            logger.info("ValidationAgent: repair attempt.")
            result = await self._agent.run(repair_prompt)
            logger.info(
                "ValidationAgent: repair success — is_accurate={}, confidence={:.2f}",
                result.output.is_accurate,
                result.output.confidence,
            )
            return result.output

        except (ValidationError, Exception) as second_exc:
            logger.error(
                "ValidationAgent: both attempts failed. Returning safe fallback result. Error: {}",
                second_exc,
            )
            return ValidationResult(
                is_grounded=False,
                is_accurate=False,
                final_answer=info_output.raw_answer,
                confidence=0.0,
                issues=[
                    "Validation agent failed to produce a structured response after two attempts.",
                    f"Final error: {type(second_exc).__name__}: {str(second_exc)[:200]}",
                ],
            )

    def validate(
        self,
        user_question: str,
        info_output: InformationAgentOutput,
        source_chunks: Optional[List[dict]] = None,
    ) -> ValidationResult:
        """Synchronous wrapper around validate_async."""
        return asyncio.get_event_loop().run_until_complete(
            self.validate_async(user_question, info_output, source_chunks)
        )

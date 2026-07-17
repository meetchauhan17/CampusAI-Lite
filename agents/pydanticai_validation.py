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

        # Safely strip — response.text can be None on safety-filtered responses
        raw_text = response.text or ""
        text = raw_text.strip()

        # Clean markdown code block wraps to prevent PydanticAI ValidationErrors
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].endswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        text = text.strip("`").strip()

        # Guard: PydanticAI raises "model output must contain either output text
        # or tool calls, these cannot both be empty" if TextPart.content is ""
        # (happens on Gemini safety blocks, watsonx empty completions, or None)
        if not text:
            logger.warning(
                "[ValidationAgent FunctionModel] LLM returned empty/None text via provider '{}'. "
                "Injecting fallback JSON so PydanticAI does not crash.",
                response.provider_used,
            )
            text = json.dumps({
                "is_grounded": False,
                "is_accurate": False,
                "final_answer": "The LLM returned an empty response and could not validate the answer.",
                "confidence": 0.0,
                "issues": [f"LLM provider '{response.provider_used}' returned empty/None text (possible safety filter or quota issue)."]
            })

        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(function=campus_llm_function, model_name="campus-llm-bridge")


# ────────────────────────────────────────────────────────────────────────────
# System prompt for the Validation Agent
# ────────────────────────────────────────────────────────────────────────────

VALIDATION_SYSTEM_PROMPT = """
You are a STRICT university information fact-checker.

Your ONLY information source is the numbered retrieved document chunks provided in the user message.
You must NOT use any prior knowledge, assumptions, or external information.

STEP-BY-STEP INSTRUCTIONS:
1. Read the user's original question and the raw_answer from the Information Agent.
2. Identify every concrete factual claim in raw_answer: dates, times, hall/block numbers, fees, amounts, deadlines, book limits, etc.
3. For EACH claim, compare it to the retrieved chunks:
   - Verify if the claim is factually supported by the chunks. (Phrasing can vary, e.g., "at 10:30 AM" is supported by "Time: 10:30 AM").
   - If a claim is supported, DO NOT list it in the "issues" array.
   - If a claim is incorrect, contradicts the chunks, or is missing from the chunks, list it as an issue in the "issues" array. Format the issue clearly, e.g., "Claim '<claim>' contradicts retrieved chunk..." or "Claim '<claim>' not found in chunks...".
4. The "issues" array must ONLY contain actual errors, contradictions, or missing facts. It MUST be completely empty `[]` if all claims are correct. Never include positive notes like "Claim X is supported" in the "issues" array.
5. Set is_grounded=True if all factual claims in raw_answer are supported by the chunks.
6. Set is_accurate=True if and only if is_grounded is True AND there are no contradictions, unverified claims, or discrepancies. If there are any discrepancies, set is_accurate=False.
7. If the raw_answer has no issues and is completely accurate, set is_accurate=True, is_grounded=True, and issues=[].
8. Set confidence based on verification:
   - All claims verified and accurate: 0.90 to 1.00
   - Most claims verified and accurate (>=75%): 0.60 to 0.89
   - Major contradictions or unverified claims: 0.00 to 0.59
9. In final_answer: if the raw_answer is accurate, return it as-is. If there are errors or unverified claims, correct them using ONLY the facts from the chunks.

You MUST respond with a JSON object matching this exact schema and nothing else:
{
  "is_grounded": <bool>,
  "is_accurate": <bool>,
  "final_answer": "<string>",
  "confidence": <float between 0.0 and 1.0>,
  "issues": ["<issue 1>", "<issue 2>", ...]
}

Do not include markdown code fences, prose, or any other text outside the JSON object.
Do not invent facts. Do not trust the raw_answer without verifying against the chunks.
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
        if source_chunks:
            chunks_text = "\n\n".join(
                f"[Chunk {i+1} | Source: {c.get('source_file', 'unknown')} | Category: {c.get('category', 'n/a')}]\n{c.get('text', '')}"
                for i, c in enumerate(source_chunks)
            )
        else:
            chunks_text = "No source chunks were retrieved."

        return (
            f"Original user question:\n{user_question}\n\n"
            f"Raw answer from Information Agent:\n{info_output.raw_answer}\n\n"
            f"Cited sources: {', '.join(info_output.sources) or 'none'}\n\n"
            f"Retrieved document chunks (your ONLY allowed information source):\n{chunks_text}\n\n"
            f"TASK: Verify that every factual claim (dates, times, halls, amounts, limits) in the raw answer is factually supported by the numbered chunks above. "
            f"Do NOT require identical phrasing (e.g., 'from 10:30 AM to 01:00 PM' is supported by 'Time: 10:30 AM to 01:00 PM', and 'in Block A, Hall 1-3' is supported by 'Venues: Block A, Hall 1-3'). "
            f"Only flag an issue if the core fact is incorrect, contradicts the sources, or is completely missing from the chunks."
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
        try:
            return asyncio.run(
                self.validate_async(user_question, info_output, source_chunks)
            )
        except RuntimeError as e:
            if "already running" in str(e) or "cannot be called from a running event loop" in str(e):
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(
                        lambda: asyncio.run(self.validate_async(user_question, info_output, source_chunks))
                    ).result()
            raise

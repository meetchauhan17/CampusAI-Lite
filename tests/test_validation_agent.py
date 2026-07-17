"""
tests/test_validation_agent.py

Two test cases for ValidationAgent:
  1. Well-grounded answer  → is_accurate=True, empty issues list.
  2. Contradictory answer  → is_accurate=False, non-empty issues list.

The LLMProvider is mocked so no real API calls are made.
The mock function produces a hard-coded JSON string that PydanticAI parses into
ValidationResult via its output_type machinery.
"""
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from core.schemas import InformationAgentOutput, ValidationResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_provider_mock(json_response: str) -> MagicMock:
    """Return a MagicMock LLMProvider whose generate_async yields json_response."""
    from core.llm_provider import LLMResponse
    mock = MagicMock()
    mock.generate_async = AsyncMock(
        return_value=LLMResponse(
            text=json_response,
            provider_used="mock",
            latency_ms=1.0,
            tokens_used=10,
            fallback_triggered=False,
        )
    )
    return mock


SOURCE_CHUNKS = [
    {
        "source_file": "fee_structure.txt",
        "category": "fees",
        "text": (
            "B.E. / B.Tech (Undergraduate) Tuition Fees:\n"
            "- Tuition Fee (per semester): INR 45,000\n"
            "- Enrollment Fee (one-time, at admission): INR 1,500\n"
            "- Total Estimated First Semester Fee: INR 54,700"
        ),
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationAgent(unittest.IsolatedAsyncioTestCase):

    async def test_grounded_answer_passes(self):
        """
        When the raw_answer correctly reflects the sources, the ValidationAgent
        should confirm is_grounded=True, is_accurate=True, and return an empty issues list.
        """
        from agents.pydanticai_validation import ValidationAgent

        mock_json = json.dumps({
            "is_grounded": True,
            "is_accurate": True,
            "final_answer": "The B.Tech tuition fee per semester is INR 45,000. The total estimated first semester fee (including enrollment and exam fees) is INR 54,700.",
            "confidence": 0.95,
            "issues": [],
        })
        provider = _make_provider_mock(mock_json)
        agent = ValidationAgent(llm_provider=provider)

        info_output = InformationAgentOutput(
            raw_answer="The B.Tech tuition fee per semester is INR 45,000.",
            sources=["fee_structure.txt"],
            category="fees",
        )

        result = await agent.validate_async(
            user_question="What is the B.Tech tuition fee per semester?",
            info_output=info_output,
            source_chunks=SOURCE_CHUNKS,
        )

        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_grounded)
        self.assertTrue(result.is_accurate)
        self.assertEqual(result.issues, [])
        self.assertGreater(result.confidence, 0.5)
        self.assertIn("45,000", result.final_answer)

    async def test_contradictory_answer_flagged(self):
        """
        When the raw_answer contradicts the sources (wrong fee amount), the
        ValidationAgent should flag is_accurate=False and populate the issues list.
        """
        from agents.pydanticai_validation import ValidationAgent

        mock_json = json.dumps({
            "is_grounded": False,
            "is_accurate": False,
            "final_answer": "According to the official fee structure, the B.Tech tuition fee is INR 45,000 per semester, not INR 80,000 as stated.",
            "confidence": 0.85,
            "issues": [
                "The raw answer states INR 80,000 per semester but the source document clearly states INR 45,000 per semester."
            ],
        })
        provider = _make_provider_mock(mock_json)
        agent = ValidationAgent(llm_provider=provider)

        info_output = InformationAgentOutput(
            raw_answer="The B.Tech tuition fee per semester is INR 80,000.",  # WRONG
            sources=["fee_structure.txt"],
            category="fees",
        )

        result = await agent.validate_async(
            user_question="What is the B.Tech tuition fee per semester?",
            info_output=info_output,
            source_chunks=SOURCE_CHUNKS,
        )

        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.is_accurate)
        self.assertGreater(len(result.issues), 0)
        # The corrected answer should mention the real amount
        self.assertIn("45,000", result.final_answer)
        # At least one issue should call out the wrong figure
        issues_text = " ".join(result.issues)
        self.assertIn("80,000", issues_text)


if __name__ == "__main__":
    unittest.main()

import asyncio
import json
from typing import Any, AsyncGenerator, Dict, List, Optional
from pydantic import BaseModel, Field

from beeai_framework import BeeAgent, UnconstrainedMemory, Message, AssistantMessage, Role, Tool
from beeai_framework.agents.bee.agent import BeeInput
from beeai_framework.backend.chat import ChatModel, ChatModelInput, ChatModelOutput, ChatModelUsage, RunContext
from beeai_framework.agents.types import BeeRunInput, AgentMeta

from config.settings import Settings
from core.logger import logger
from core.llm_provider import LLMProvider

# ─────────────────────────────────────────────────────────────────────────────
# 1. Custom ChatModel Backend for BeeAI
# ─────────────────────────────────────────────────────────────────────────────

class CampusAIBeeChatModel(ChatModel):
    """
    Custom BeeAI ChatModel backend that routes messages to our unified LLMProvider.
    Allows BeeAI agents to run Granite / Groq / Gemini in the same failover chain.
    """
    def __init__(self, provider: LLMProvider):
        super().__init__()
        self.provider = provider

    @property
    def model_id(self) -> str:
        return "campus-ai-provider"

    @property
    def provider_id(self) -> str:
        return "custom"

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        system_prompt = None
        user_prompt = ""
        
        for m in input.messages:
            role = m.role
            text = m.text
            if role == Role.SYSTEM:
                system_prompt = text
            else:
                user_prompt += f"\n[{role}]: {text}" if user_prompt else text

        response = await self.provider.generate_async(
            prompt=user_prompt,
            system=system_prompt
        )

        return ChatModelOutput(
            messages=[AssistantMessage(response.text)],
            usage=ChatModelUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=response.tokens_used
            )
        )

    async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput, None]:
        # Simple stream implementation: yield the complete output in one chunk
        yield await self._create(input, run)

    async def _create_structure(
        self,
        input: Any,
        run: RunContext
    ) -> Any:
        # Delegate to the superclass default implementation
        return await super()._create_structure(input, run)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Custom BeeAI Tool Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SearchInputSchema(BaseModel):
    query: str = Field(description="Search query to execute against university documents.")

class UniversityInfoSearchToolBee(Tool):
    """
    BeeAI-native wrapper for the UniversityInfoSearchTool.
    Exposes input_schema for parameter validation and runs the custom search.
    """
    name = "UniversityInfoSearchTool"
    description = (
        "Search university documents (exams, fees, library, hostel, academic calendar) "
        "to answer student queries. Input should be a specific search query string."
    )
    input_schema = SearchInputSchema

    def _run(self, input: SearchInputSchema, options: Optional[dict] = None) -> "StringToolOutput":
        from beeai_framework.tools.tool import StringToolOutput
        from tools.university_search_tool import _execute_university_search
        res = _execute_university_search(input.query)
        return StringToolOutput(json.dumps(res, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Two-Agent PoC Execution
# ─────────────────────────────────────────────────────────────────────────────

async def run_beeai_poc_async(
    user_question: str,
    settings: Optional[Settings] = None,
    provider: Optional[LLMProvider] = None
) -> dict:
    """
    Async implementation of the two-agent BeeAI flow:
    Router agent (classifies category) -> Responder agent (retrieves and answers).
    """
    if settings is None:
        settings = Settings()
    if provider is None:
        provider = LLMProvider(settings)

    logger.info("[BeeAI] Starting PoC for question: '{}'", user_question[:80])

    chat_model = CampusAIBeeChatModel(provider)
    search_tool = UniversityInfoSearchToolBee()

    # ── Agent 1: Router Agent ───────────────────────────────────────────────
    router_agent = BeeAgent(
        BeeInput(
            llm=chat_model,
            tools=[],
            memory=UnconstrainedMemory(),
            meta=AgentMeta(name="Router", description="Router Agent", tools=[])
        )
    )

    router_prompt = (
        f"Classify the student's question into exactly one category from: "
        f"exams, fees, library, hostel, academic-calendar, or general.\n"
        f"Return ONLY the category name and nothing else.\n\n"
        f"Question: {user_question}"
    )

    router_res = await router_agent.run(BeeRunInput(prompt=router_prompt))
    category = router_res.result.text.strip().lower().strip("'\"")

    # ── Agent 2: Responder Agent ────────────────────────────────────────────
    # Pre-fetch chunks so the agent has the official documents directly in context
    from tools.university_search_tool import _execute_university_search
    search_res = _execute_university_search(user_question)
    chunks = search_res.get("answer_chunks", [])
    sources = list(set(c.get("source_file") for c in chunks if c.get("source_file")))

    responder_agent = BeeAgent(
        BeeInput(
            llm=chat_model,
            tools=[search_tool],
            memory=UnconstrainedMemory(),
            meta=AgentMeta(name="Responder", description="Responder Agent", tools=[search_tool])
        )
    )

    responder_prompt = (
        f"You are a university responder assistant. Answer the user question: {user_question}\n"
        f"Category of question: {category}\n"
        f"Here are the official document chunks retrieved from the university files:\n"
        f"{json.dumps(chunks, indent=2)}\n\n"
        f"You MUST use the university document chunks above to answer the question. Do not make up facts."
    )

    responder_res = await responder_agent.run(BeeRunInput(prompt=responder_prompt))
    answer = responder_res.result.text

    logger.info("[BeeAI] PoC finished. Category={}, Sources={}", category, sources)

    return {
        "category": category,
        "answer": answer,
        "sources": sources
    }


def run_beeai_poc(
    user_question: str,
    settings: Optional[Settings] = None,
    provider: Optional[LLMProvider] = None
) -> dict:
    """
    Synchronous entry point for the BeeAI PoC workflow.
    """
    try:
        # Use asyncio.run for robust, thread-safe event loop execution
        return asyncio.run(
            run_beeai_poc_async(user_question, settings, provider)
        )
    except Exception as exc:
        logger.error("[BeeAI] PoC execution failed: {}", exc)
        return {
            "category": "general",
            "answer": f"System error occurred: {str(exc)}",
            "sources": []
        }


# ─────────────────────────────────────────────────────────────────────────────
# Verification block
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLE_QUESTIONS = [
        "When is the Artificial Intelligence exam and what hall do I report to?",
        "How much is the B.Tech tuition fee and what happens if I pay late?",
        "How many books can a postgraduate student borrow from the library?",
    ]

    settings = Settings()
    provider = LLMProvider(settings)

    print("--- BeeAI Two-Agent PoC Verification ---")
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(SAMPLE_QUESTIONS)}] Question: {question}")
        print("="*80)

        result = run_beeai_poc(question, settings, provider)
        print("Result Dict:")
        print(f"  Category Detected: {result['category']}")
        print(f"  Sources Cited:     {result['sources']}")
        print(f"  Answer Snippet:    {result['answer'][:150]}...")
        print()

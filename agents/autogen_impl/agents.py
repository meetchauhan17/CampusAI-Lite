"""
agents/autogen_impl/agents.py
==============================
AG2 (AutoGen) custom model client + agent factory.

Key design decisions:
  * llm_config contains ONLY plain JSON-serializable data so AG2's deepcopy
    of it in ConversableAgent.__init__ never encounters un-picklable objects
    (httpx CookieJar, RLock, etc.).
  * The live LLMProvider is held in a module-level registry dict keyed by a
    short string ID that IS safe to deep-copy.
  * CampusAIAutoGenClient.__init__ looks up the provider from that registry
    rather than receiving it through config.
"""

import json
from typing import Any, List, Dict, Optional

from autogen import ModelClient, ConversableAgent
from config.settings import Settings
from core.llm_provider import LLMProvider
from core.logger import logger

# ─────────────────────────────────────────────────────────────────────────────
# Module-level provider registry
# AG2 deep-copies llm_config but never touches this dict, so we can safely
# store live provider instances here.
# ─────────────────────────────────────────────────────────────────────────────
_PROVIDER_REGISTRY: Dict[str, LLMProvider] = {}

_DEFAULT_PROVIDER_KEY = "default"


def register_provider(provider: LLMProvider, key: str = _DEFAULT_PROVIDER_KEY) -> None:
    """Store a provider instance in the registry so CampusAIAutoGenClient can find it."""
    _PROVIDER_REGISTRY[key] = provider


def get_provider(key: str = _DEFAULT_PROVIDER_KEY) -> LLMProvider:
    """Retrieve the registered provider; raises if not set."""
    if key not in _PROVIDER_REGISTRY:
        raise RuntimeError(
            f"No LLMProvider registered under key '{key}'. "
            "Call register_provider() before constructing agents."
        )
    return _PROVIDER_REGISTRY[key]


# ─────────────────────────────────────────────────────────────────────────────
# AutoGen response helper
# ─────────────────────────────────────────────────────────────────────────────

class AutoGenResponseObject(dict):
    """
    A dict subclass that also supports attribute access.
    Satisfies both Pydantic dict checks and AutoGen's attribute-style access.
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.__dict__ = self


# ─────────────────────────────────────────────────────────────────────────────
# Custom AG2 ModelClient
# ─────────────────────────────────────────────────────────────────────────────

class CampusAIAutoGenClient(ModelClient):
    """
    Custom ModelClient that routes AG2 LLM calls through our LLMProvider.

    AG2 calls:  CampusAIAutoGenClient(config, **kwargs)
    where `config` is the entry from llm_config["config_list"].
    We deliberately do NOT store a provider reference inside config; instead
    we look it up from _PROVIDER_REGISTRY using the provider_key field.
    """

    def __init__(self, config: Dict[str, Any], **kwargs: Any):
        self.model_name = config.get("model", "campus-ai-provider")
        provider_key = config.get("provider_key", _DEFAULT_PROVIDER_KEY)
        self._provider = get_provider(provider_key)
        logger.debug(
            "[CampusAIAutoGenClient] Initialized with model='{}' provider_key='{}'",
            self.model_name, provider_key,
        )

    # ── AG2 ModelClient protocol ──────────────────────────────────────────────

    def create(self, params: Dict[str, Any]) -> AutoGenResponseObject:
        messages = params.get("messages", [])

        # Check if the last message is a tool response (so we don't loop)
        last_msg = messages[-1] if messages else {}
        is_tool_response = (
            last_msg.get("role") == "tool"
            or "tool_responses" in last_msg
            or any("tool" in str(msg.get("role", "")).lower() for msg in messages[-2:])
        )

        # If tools are declared and we haven't yet executed one, emit a tool call
        if "tools" in params and params["tools"] and not is_tool_response:
            user_question = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_question = msg.get("content", "")
                    break
            if not user_question:
                user_question = last_msg.get("content", "")

            tool_call = AutoGenResponseObject(
                id="call_search_1",
                type="function",
                function=AutoGenResponseObject(
                    name="search_university_info",
                    arguments=json.dumps({"query": user_question}),
                ),
            )
            choice = AutoGenResponseObject(
                message=AutoGenResponseObject(
                    content=None,
                    role="assistant",
                    tool_calls=[tool_call],
                    function_call=None,
                )
            )
            return AutoGenResponseObject(
                choices=[choice],
                model=self.model_name,
                usage=AutoGenResponseObject(
                    prompt_tokens=0, completion_tokens=0, total_tokens=0
                ),
            )

        # Standard text generation
        system_prompt: Optional[str] = None
        user_prompt = ""
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "") or ""
            if role == "system":
                system_prompt = content
            else:
                user_prompt += f"\n[{role}]: {content}"

        temperature = params.get("temperature", 0.3)
        max_tokens = params.get("max_tokens", 1024)

        logger.info("[CampusAIAutoGenClient] Generating response (model={}).", self.model_name)
        response = self._provider.generate(
            prompt=user_prompt.strip(),
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = AutoGenResponseObject(
            message=AutoGenResponseObject(
                content=response.text,
                role="assistant",
                tool_calls=None,
                function_call=None,
            )
        )
        return AutoGenResponseObject(
            choices=[choice],
            model=self.model_name,
            usage=AutoGenResponseObject(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=response.tokens_used or 0,
            ),
        )

    def message_retrieval(self, response: AutoGenResponseObject) -> List[str]:
        choices = getattr(response, "choices", [])
        return [
            choice.message.content
            for choice in choices
            if getattr(choice.message, "content", None) is not None
        ]

    def cost(self, response: AutoGenResponseObject) -> float:  # noqa: D102
        return 0.0

    @staticmethod
    def get_usage(response: AutoGenResponseObject) -> Dict[str, int]:  # noqa: D102
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────────────────

def build_autogen_agents(
    provider: LLMProvider, settings: Settings
) -> Dict[str, ConversableAgent]:
    """
    Register the provider singleton then build all four ConversableAgents.

    llm_config contains ONLY plain, JSON-serializable data so AG2's internal
    deepcopy never encounters un-picklable objects.
    """
    # 1. Store provider in registry BEFORE constructing any agent
    register_provider(provider)

    # 2. Pure-JSON llm_config — no live objects anywhere in this dict.
    # Use model_client_cls (string name) so AG2 routes to our custom client;
    # do NOT use api_type — AG2 validates it against a hard-coded enum and
    # 'campusai' is not in the list.
    llm_config: Dict[str, Any] = {
        "config_list": [
            {
                "model": "campus-ai-provider",
                "model_client_cls": "CampusAIAutoGenClient",
                "provider_key": _DEFAULT_PROVIDER_KEY,
                "temperature": 0.3,
                "max_tokens": 1024,
            }
        ]
    }

    # 3. Planner
    planner = ConversableAgent(
        name="PlannerAgent",
        system_message=(
            "You are a university query planner. Analyze the student's question and produce a plan.\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "sub_tasks": ["list of tasks"],\n'
            '  "category": "exams/fees/library/hostel/academic-calendar/general",\n'
            '  "requires_tool": true/false\n'
            "}"
        ),
        llm_config=llm_config,
        code_execution_config=False,
        human_input_mode="NEVER",
    )
    planner.register_model_client(model_client_cls=CampusAIAutoGenClient)

    # 4. Information / Retrieval
    information = ConversableAgent(
        name="InformationAgent",
        system_message=(
            "You are a university information retriever assistant.\n"
            "Analyze the student's question, execute the search tool to find facts, and answer the question.\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "raw_answer": "your answer here",\n'
            '  "sources": ["list of source files referenced"],\n'
            '  "category": "category matching the planner\'s category"\n'
            "}"
        ),
        llm_config=llm_config,
        code_execution_config=False,
        human_input_mode="NEVER",
    )
    information.register_model_client(model_client_cls=CampusAIAutoGenClient)

    # 5. Validator
    validator = ConversableAgent(
        name="ValidationAgent",
        system_message="You are a fact checker. Read the question, chunks, and answer and validate them.",
        llm_config=llm_config,
        code_execution_config=False,
        human_input_mode="NEVER",
    )
    validator.register_model_client(model_client_cls=CampusAIAutoGenClient)

    # 6. UserProxy (no LLM, just executes tools)
    user_proxy = ConversableAgent(
        name="user_proxy",
        llm_config=False,
        code_execution_config=False,
        human_input_mode="NEVER",
    )

    return {
        "planner": planner,
        "information": information,
        "validator": validator,
        "user_proxy": user_proxy,
    }

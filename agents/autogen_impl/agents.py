import json
from typing import Any, List, Dict, Optional
from autogen import ModelClient, ConversableAgent
from config.settings import Settings
from core.llm_provider import LLMProvider

class DictNamespace:
    """
    A namespace object that supports both attribute access and dictionary-like access
    (keys, get, indexing, and iteration), making it fully compatible with AutoGen's internal structures.
    """
    def __init__(self, **kwargs: Any):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __iter__(self):
        return iter(self.__dict__.items())

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item, default=None):
        return getattr(self, item, default)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return f"DictNamespace({self.__dict__})"


class CampusAIAutoGenClient(ModelClient):
    """
    A custom ModelClient that routes AutoGen LLM calls to our LLMProvider.
    Supports tool calling by simulating function call responses when tools are declared.
    """
    def __init__(self, config: Dict[str, Any], **kwargs: Any):
        self.config = config
        self.provider = config.get("provider_instance")
        if not self.provider:
            raise ValueError("provider_instance must be supplied in config")

    def create(self, params: Dict[str, Any]) -> DictNamespace:
        messages = params.get("messages", [])
        
        # Check if the last message was a tool response
        last_msg = messages[-1] if messages else {}
        is_tool_response = (
            last_msg.get("role") == "tool" or
            "tool_responses" in last_msg or
            any("tool" in str(msg.get("role")).lower() for msg in messages[-2:])
        )

        # Trigger tool call if search tools are defined and we haven't executed it yet
        if "tools" in params and params["tools"] and not is_tool_response:
            # Locate the original user question in the history
            user_question = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_question = msg.get("content", "")
                    break
            if not user_question:
                user_question = last_msg.get("content", "")

            # Simulate tool call structure for AutoGen execution
            tool_call = DictNamespace(
                id="call_search_1",
                type="function",
                function=DictNamespace(
                    name="search_university_info",
                    arguments=json.dumps({"query": user_question})
                )
            )
            choice = DictNamespace(
                message=DictNamespace(
                    content=None,
                    role="assistant",
                    tool_calls=[tool_call],
                    function_call=None
                )
            )
            return DictNamespace(
                choices=[choice],
                model="campus-ai-provider",
                usage=DictNamespace(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0
                )
            )

        # Otherwise, perform standard text generation
        system_prompt = None
        user_prompt = ""
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "system":
                system_prompt = content
            else:
                user_prompt += f"\n[{role}]: {content}"

        response = self.provider.generate(
            prompt=user_prompt,
            system=system_prompt
        )

        choice = DictNamespace(
            message=DictNamespace(
                content=response.text,
                role="assistant",
                tool_calls=None,
                function_call=None
            )
        )
        return DictNamespace(
            choices=[choice],
            model="campus-ai-provider",
            usage=DictNamespace(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=response.tokens_used
            )
        )

    def message_retrieval(self, response: DictNamespace) -> List[str]:
        choices = getattr(response, "choices", [])
        return [choice.message.content for choice in choices if getattr(choice.message, "content", None) is not None]

    def cost(self, response: DictNamespace) -> float:
        return 0.0

    @staticmethod
    def get_usage(response: DictNamespace) -> Dict[str, int]:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost": 0.0}


def build_autogen_agents(provider: LLMProvider, settings: Settings) -> Dict[str, ConversableAgent]:
    """
    Builds the four agents (Planner, Information, Validation, and UserProxy)
    configured with the custom model client.
    """
    llm_config = {
        "config_list": [
            {
                "model": "campus-ai-provider",
                "model_client_cls": "CampusAIAutoGenClient",
                "provider_instance": provider
            }
        ]
    }

    # 1. PlannerAgent
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
        human_input_mode="NEVER"
    )
    planner.register_model_client(model_client_cls=CampusAIAutoGenClient)

    # 2. InformationAgent
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
        human_input_mode="NEVER"
    )
    information.register_model_client(model_client_cls=CampusAIAutoGenClient)

    # 3. ValidationAgent
    validator = ConversableAgent(
        name="ValidationAgent",
        system_message=(
            "You are a fact checker. You will read the question, chunks, and answer, and validate them."
        ),
        llm_config=llm_config,
        code_execution_config=False,
        human_input_mode="NEVER"
    )
    validator.register_model_client(model_client_cls=CampusAIAutoGenClient)

    # 4. user_proxy (Executor)
    user_proxy = ConversableAgent(
        name="user_proxy",
        llm_config=False,  # No LLM for proxy
        code_execution_config=False,
        human_input_mode="NEVER"
    )

    return {
        "planner": planner,
        "information": information,
        "validator": validator,
        "user_proxy": user_proxy
    }

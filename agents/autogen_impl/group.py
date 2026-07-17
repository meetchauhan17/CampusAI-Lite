import json
import re
from typing import Any, Dict, List, Optional
from autogen import GroupChat, GroupChatManager, ConversableAgent, register_function
from config.settings import Settings
from core.logger import logger
from core.llm_provider import LLMProvider
from core.schemas import ValidationResult, InformationAgentOutput
from tools.university_search_tool import search_university_info
from agents.pydanticai_validation import ValidationAgent
from agents.autogen_impl.agents import build_autogen_agents, CampusAIAutoGenClient, register_provider, _DEFAULT_PROVIDER_KEY

# ─────────────────────────────────────────────────────────────────────────────
# JSON Extraction Helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """
    Robust JSON parser that extracts JSON blocks from potential markdown formatting.
    """
    stripped = re.sub(r"```(?:json)?", "", text).strip().strip("`")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
            
    raise ValueError(f"Could not extract JSON from text: {text[:200]}")


# ─────────────────────────────────────────────────────────────────────────────
# Custom Reply for ValidationAgent
# ─────────────────────────────────────────────────────────────────────────────

def validation_agent_custom_reply(
    recipient: ConversableAgent,
    messages: List[Dict[str, Any]],
    sender: ConversableAgent,
    config: Dict[str, Any]
) -> tuple[bool, Dict[str, Any]]:
    """
    ValidationAgent custom reply function.
    Reads conversation history, extracts the user question, information agent output,
    and retrieved search chunks, then delegates validation to the PydanticAI validation agent.
    """
    logger.info("[AutoGen] ValidationAgent executing custom fact-check reply.")

    # 1. Extract the user question
    user_question = ""
    for msg in messages:
        if msg.get("name") == "user_proxy" and msg.get("role") == "user":
            user_question = msg.get("content", "")
            break
    if not user_question and messages:
        user_question = messages[0].get("content", "")

    # 2. Extract InformationAgent output
    raw_answer = ""
    sources = []
    category = "general"
    for msg in reversed(messages):
        if msg.get("name") == "InformationAgent" and msg.get("content"):
            content = msg.get("content", "")
            try:
                data = _extract_json(content)
                raw_answer = data.get("raw_answer", content)
                sources = data.get("sources", [])
                category = data.get("category", "general")
            except Exception:
                raw_answer = content
            break

    # 3. Extract retrieved chunks from the tool execution
    chunks = []
    for msg in messages:
        # Check for tool response content in user_proxy tool execution messages
        if msg.get("role") == "tool" or (msg.get("name") == "user_proxy" and "tool_responses" in str(msg)):
            content = msg.get("content", "")
            try:
                data = json.loads(content)
                chunks = data.get("answer_chunks", [])
            except Exception:
                pass

    # Fallback to direct search if no chunks were loaded in groupchat logs
    if not chunks and user_question:
        from tools.university_search_tool import _execute_university_search
        search_res = _execute_university_search(user_question)
        chunks = search_res.get("answer_chunks", [])

    info_output = InformationAgentOutput(
        raw_answer=raw_answer or "No answer generated.",
        sources=sources,
        category=category
    )

    # 4. Delegate to PydanticAI ValidationAgent
    validation_agent: ValidationAgent = config["validation_agent_instance"]
    validation_result = validation_agent.validate(
        user_question=user_question,
        info_output=info_output,
        source_chunks=chunks
    )

    # Return serialized output with a final termination key
    reply_content = f"FINAL ANSWER:\n{validation_result.model_dump_json()}"
    return True, {"content": reply_content, "role": "assistant"}


# ─────────────────────────────────────────────────────────────────────────────
# Custom Speaker Selection State Machine
# ─────────────────────────────────────────────────────────────────────────────

def select_speaker_sequence(last_speaker: ConversableAgent, groupchat: GroupChat) -> Optional[ConversableAgent]:
    """
    Enforces a strict sequential transition graph:
    START (user_proxy) -> PlannerAgent -> InformationAgent -> ValidationAgent -> END.
    If InformationAgent outputs a tool call, routes to user_proxy to execute the tool
    before routing back to InformationAgent.
    """
    if last_speaker is None:
        return groupchat.agent_by_name("PlannerAgent")

    name = last_speaker.name

    if name == "user_proxy":
        # If user_proxy just spoke, check if it was executing a tool
        if groupchat.messages:
            last_msg = groupchat.messages[-1]
            if last_msg.get("role") == "tool" or "tool_responses" in str(last_msg):
                return groupchat.agent_by_name("InformationAgent")
        # Otherwise it was the initial question
        return groupchat.agent_by_name("PlannerAgent")

    elif name == "PlannerAgent":
        return groupchat.agent_by_name("InformationAgent")

    elif name == "InformationAgent":
        if groupchat.messages:
            last_msg = groupchat.messages[-1]
            # Check if tool calling was suggested
            if last_msg.get("tool_calls") or last_msg.get("function_call"):
                return groupchat.agent_by_name("user_proxy")
        return groupchat.agent_by_name("ValidationAgent")

    elif name == "ValidationAgent":
        return None  # Terminate chat

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public Pipeline API
# ─────────────────────────────────────────────────────────────────────────────

def run_autogen_pipeline(
    user_question: str,
    settings: Optional[Settings] = None,
    provider: Optional[LLMProvider] = None
) -> ValidationResult:
    """
    Run the full sequential AG2 (AutoGen) GroupChat pipeline and return a
    guaranteed-schema ValidationResult.
    """
    if settings is None:
        settings = Settings()
    if provider is None:
        provider = LLMProvider(settings)

    logger.info("[AutoGen] Starting pipeline for question: '{}'", user_question[:80])

    try:
        # Build agents
        agents = build_autogen_agents(provider, settings)
        planner = agents["planner"]
        information = agents["information"]
        validator = agents["validator"]
        user_proxy = agents["user_proxy"]

        # Register search tool
        register_function(
            search_university_info,
            caller=information,
            executor=user_proxy,
            name="search_university_info",
            description="Search university documents (exams, fees, library, hostel, academic calendar) to answer student queries."
        )

        # Re-register client on all LLM agents to ensure it is active right before execution
        planner.register_model_client(model_client_cls=CampusAIAutoGenClient)
        information.register_model_client(model_client_cls=CampusAIAutoGenClient)
        validator.register_model_client(model_client_cls=CampusAIAutoGenClient)

        # Register custom validation reply
        validation_agent = ValidationAgent(provider)
        validator.register_reply(
            [ConversableAgent, None],
            reply_func=validation_agent_custom_reply,
            position=0,
            config={"validation_agent_instance": validation_agent}
        )

        # Set up group chat
        groupchat = GroupChat(
            agents=[planner, information, validator, user_proxy],
            messages=[],
            max_round=10,
            speaker_selection_method=select_speaker_sequence
        )

        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config={
                "config_list": [
                    {
                        "model": "campus-ai-provider",
                        "model_client_cls": "CampusAIAutoGenClient",
                        "provider_key": _DEFAULT_PROVIDER_KEY,
                        "temperature": 0.3,
                        "max_tokens": 1024,
                    }
                ]
            },
        )
        manager.register_model_client(model_client_cls=CampusAIAutoGenClient)

        # Initiate chat
        user_proxy.initiate_chat(
            manager,
            message=user_question,
            clear_history=True
        )

        # Retrieve the final message from the ValidationAgent containing the ValidationResult
        final_msg = ""
        for msg in reversed(groupchat.messages):
            if msg.get("name") == "ValidationAgent" and "FINAL ANSWER:" in str(msg.get("content")):
                final_msg = msg.get("content", "")
                break

        if not final_msg:
            # Fallback to the last message overall
            final_msg = groupchat.messages[-1].get("content", "") if groupchat.messages else ""

        # Parse final validated response
        if "FINAL ANSWER:" in final_msg:
            json_text = final_msg.split("FINAL ANSWER:")[-1].strip()
            data = _extract_json(json_text)
            return ValidationResult(
                is_grounded=bool(data.get("is_grounded", False)),
                is_accurate=bool(data.get("is_accurate", False)),
                final_answer=str(data.get("final_answer", "")),
                confidence=float(data.get("confidence", 0.0)),
                issues=list(data.get("issues", []))
            )
        else:
            # Attempt to parse directly
            data = _extract_json(final_msg)
            return ValidationResult(**data)

    except Exception as exc:
        import traceback
        logger.exception("[AutoGen] Pipeline failed with traceback:")
        return ValidationResult(
            is_grounded=False,
            is_accurate=False,
            final_answer=f"AutoGen pipeline error: {type(exc).__name__}: {str(exc)}",
            confidence=0.0,
            issues=[f"{type(exc).__name__}: {str(exc)}", traceback.format_exc()]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Verification block
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint
    from agents.crewai_impl.crew import run_crewai_pipeline
    from agents.langgraph_impl.graph import run_langgraph_pipeline

    SAMPLE_QUESTIONS = [
        "When is the Artificial Intelligence exam and what hall do I report to?",
        "How much is the B.Tech tuition fee and what happens if I pay late?",
        "How many books can a postgraduate student borrow from the library?",
    ]

    settings = Settings()
    provider = LLMProvider(settings)

    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(SAMPLE_QUESTIONS)}] Question: {question}")
        print("="*80)

        print("\n--- Running AutoGen Pipeline ---")
        ag_res = run_autogen_pipeline(question, settings, provider)

        print("\n--- Running LangGraph Pipeline ---")
        lg_res = run_langgraph_pipeline(question, settings, provider)

        print("\n--- Running CrewAI Pipeline ---")
        cr_res = run_crewai_pipeline(question, settings, provider)

        print("\nThree-Way Pipeline Comparison:")
        print(f"{'Field':<20} | {'AutoGen':<22} | {'LangGraph':<22} | {'CrewAI':<22}")
        print("-" * 96)
        print(f"{'Is Accurate':<20} | {str(ag_res.is_accurate):<22} | {str(lg_res.is_accurate):<22} | {str(cr_res.is_accurate):<22}")
        print(f"{'Is Grounded':<20} | {str(ag_res.is_grounded):<22} | {str(lg_res.is_grounded):<22} | {str(cr_res.is_grounded):<22}")
        print(f"{'Confidence':<20} | {f'{ag_res.confidence:.2f}':<22} | {f'{lg_res.confidence:.2f}':<22} | {f'{cr_res.confidence:.2f}':<22}")
        print(f"{'Issues count':<20} | {len(ag_res.issues):<22} | {len(lg_res.issues):<22} | {len(cr_res.issues):<22}")
        print(f"{'Answer snippet':<20} | {ag_res.final_answer[:20]:<22} | {lg_res.final_answer[:20]:<22} | {cr_res.final_answer[:20]:<22}")
        print()

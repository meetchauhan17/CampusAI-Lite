import time
import json
import re
import asyncio
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from core.logger import logger
from core.llm_provider import LLMProvider
from core.schemas import PlannerOutput, InformationAgentOutput, ValidationResult
from tools.university_search_tool import _execute_university_search
from agents.pydanticai_validation import ValidationAgent
from config.settings import Settings

# ─────────────────────────────────────────────────────────────────────────────
# State definition
# ─────────────────────────────────────────────────────────────────────────────

class GraphState(TypedDict):
    """
    State representing the data flow within the LangGraph workflow.
    """
    user_question: str
    planner_output: Optional[PlannerOutput]
    retrieved_chunks: List[dict]
    information_output: Optional[InformationAgentOutput]
    validation_result: Optional[ValidationResult]
    error: Optional[str]
    retry_count: int
    feedback: Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# JSON Helper
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
# LangGraph Pipeline & Nodes
# ─────────────────────────────────────────────────────────────────────────────

class LangGraphPipeline:
    """
    Capsule class enclosing the node functions for the LangGraph StateGraph,
    binding them to the shared LLMProvider and ValidationAgent.
    """
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.validation_agent = ValidationAgent(provider)

    async def plan_node(self, state: GraphState) -> dict:
        """
        Planner Node: Determines execution sub-tasks, category, and tool requirement.
        """
        start_time = time.perf_counter()
        logger.info("[LangGraph] Entering 'plan' node.")

        system_prompt = (
            "You are a university query planner. Analyze the student's question and produce a plan.\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "sub_tasks": ["list of tasks"],\n'
            '  "category": "exams/fees/library/hostel/academic-calendar/general",\n'
            '  "requires_tool": true/false\n'
            "}"
        )

        error_msg = None
        try:
            response = await self.provider.generate_async(
                prompt=state["user_question"],
                system=system_prompt
            )
            data = _extract_json(response.text)
            planner_output = PlannerOutput(
                sub_tasks=data.get("sub_tasks", []),
                category=data.get("category", "general"),
                requires_tool=bool(data.get("requires_tool", False))
            )
            logger.debug("[LangGraph][plan] category={} requires_tool={}",
                         planner_output.category, planner_output.requires_tool)
        except Exception as e:
            logger.exception("[LangGraph][plan] Failed to parse planner output; using fallback.")
            error_msg = f"plan node: {type(e).__name__}: {e}"
            planner_output = PlannerOutput(
                sub_tasks=[f"Retrieve info for: {state['user_question']}"],
                category="general",
                requires_tool=True
            )

        elapsed = (time.perf_counter() - start_time) * 1000.0
        logger.info("[LangGraph] Node 'plan' completed in {:.2f}ms", elapsed)

        result = {"planner_output": planner_output}
        if error_msg:
            result["error"] = error_msg
        return result

    async def retrieve_and_answer_node(self, state: GraphState) -> dict:
        """
        Retrieval and Answer Node: Queries the custom tool and drafts the answer.
        Handles correction feedback loop.
        """
        start_time = time.perf_counter()
        logger.info("[LangGraph] Entering 'retrieve_and_answer' node.")

        planner_output = state.get("planner_output")
        chunks = state.get("retrieved_chunks", [])

        # Always search the knowledge base when we have no cached chunks.
        # Do NOT gate on requires_tool — the planner LLM sometimes returns False
        # for factual questions, which causes the retriever to skip ChromaDB and
        # produce hallucinated "no data" answers.
        if not chunks:
            logger.info("[LangGraph] Calling UniversityInfoSearchTool (chunks empty — always search).")
            search_res = _execute_university_search(state["user_question"])
            chunks = search_res.get("answer_chunks", [])
            logger.info("[LangGraph] Search returned {} chunks for category: {}",
                        len(chunks), search_res.get("category_detected", "unknown"))

        system_prompt = (
            "You are a university information retriever assistant.\n"
            "Answer the student's question based strictly on the provided document chunks.\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "raw_answer": "your answer here",\n'
            '  "sources": ["list of source files referenced"],\n'
            '  "category": "category matching the planner\'s category"\n'
            "}"
        )

        user_prompt = f"Question: {state['user_question']}\n\nDocument chunks:\n"
        for c in chunks:
            user_prompt += f"\n- Source: {c.get('source_file')}\n{c.get('text')}\n"

        # Append validation feedback if present
        feedback = state.get("feedback")
        if feedback:
            logger.info("[LangGraph] Re-prompting retrieval with feedback: {}", feedback)
            user_prompt += (
                f"\n\nCRITICAL - Previous Answer Validation Failed:\n{feedback}\n"
                f"Please update the raw_answer to correct these inaccuracies."
            )

        raw_response_text = ""
        error_msg = None
        try:
            response = await self.provider.generate_async(
                prompt=user_prompt,
                system=system_prompt
            )
            raw_response_text = response.text
            data = _extract_json(raw_response_text)
            info_output = InformationAgentOutput(
                raw_answer=data.get("raw_answer", ""),
                sources=data.get("sources", []),
                category=data.get("category", planner_output.category if planner_output else "general")
            )
            logger.debug("[LangGraph][retrieve_and_answer] raw_answer snippet: {}",
                         info_output.raw_answer[:120])
        except Exception as e:
            logger.exception("[LangGraph][retrieve_and_answer] Failed to parse LLM answer; using raw text fallback.")
            error_msg = f"retrieve_and_answer node: {type(e).__name__}: {e}"
            info_output = InformationAgentOutput(
                raw_answer=raw_response_text or f"[Error: {e}]",
                sources=[],
                category=planner_output.category if planner_output else "general"
            )

        elapsed = (time.perf_counter() - start_time) * 1000.0
        logger.info("[LangGraph] Node 'retrieve_and_answer' completed in {:.2f}ms", elapsed)

        result = {"retrieved_chunks": chunks, "information_output": info_output}
        if error_msg:
            result["error"] = error_msg
        return result

    async def validate_node(self, state: GraphState) -> dict:
        """
        Validator Node: Delegates chunk grounding checks to the PydanticAI ValidationAgent.
        """
        start_time = time.perf_counter()
        logger.info("[LangGraph] Entering 'validate' node.")

        info_output = state.get("information_output")
        chunks = state.get("retrieved_chunks", [])

        error_msg = None
        try:
            # Call PydanticAI ValidationAgent
            validation_result = await self.validation_agent.validate_async(
                user_question=state["user_question"],
                info_output=info_output,
                source_chunks=chunks
            )
            logger.debug("[LangGraph][validate] is_accurate={} confidence={}",
                         validation_result.is_accurate, validation_result.confidence)
        except Exception as e:
            logger.exception("[LangGraph][validate] ValidationAgent raised an exception.")
            error_msg = f"validate node: {type(e).__name__}: {e}"
            validation_result = ValidationResult(
                is_grounded=False,
                is_accurate=False,
                final_answer=info_output.raw_answer if info_output else "[No answer produced]",
                confidence=0.0,
                issues=[error_msg]
            )

        new_retry_count = state.get("retry_count", 0)
        feedback = None

        if not validation_result.is_accurate:
            new_retry_count += 1
            feedback = f"Issues in answer: {', '.join(validation_result.issues)}"

        elapsed = (time.perf_counter() - start_time) * 1000.0
        logger.info("[LangGraph] Node 'validate' completed in {:.2f}ms", elapsed)

        result = {
            "validation_result": validation_result,
            "retry_count": new_retry_count,
            "feedback": feedback
        }
        if error_msg:
            result["error"] = error_msg
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Edge & Conditional Routing
# ─────────────────────────────────────────────────────────────────────────────

def check_validation(state: GraphState) -> str:
    """
    Decides routing based on validation result.
    If inaccurate and attempt limit (< 2) is not exceeded, routes back to retry.
    """
    val_res = state.get("validation_result")
    if not val_res:
        return "end"

    if not val_res.is_accurate and state.get("retry_count", 0) < 2:
        logger.info("[LangGraph] Validation failed. Retrying (Attempt {}/2).", state.get("retry_count", 0))
        return "retry"

    logger.info("[LangGraph] Validation passed or max retries reached. Exiting graph.")
    return "end"


# ─────────────────────────────────────────────────────────────────────────────
# Compilation & Public pipeline API
# ─────────────────────────────────────────────────────────────────────────────

def compile_graph(provider: LLMProvider):
    """
    Compiles the LangGraph StateGraph.
    """
    pipeline = LangGraphPipeline(provider)
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("plan", pipeline.plan_node)
    workflow.add_node("retrieve_and_answer", pipeline.retrieve_and_answer_node)
    workflow.add_node("validate", pipeline.validate_node)

    # Add Edges
    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "retrieve_and_answer")
    workflow.add_edge("retrieve_and_answer", "validate")

    # Add Conditional Edge from validate node
    workflow.add_conditional_edges(
        "validate",
        check_validation,
        {
            "retry": "retrieve_and_answer",
            "end": END
        }
    )

    return workflow.compile()


def run_langgraph_pipeline(
    user_question: str,
    settings: Optional[Settings] = None,
    provider: Optional[LLMProvider] = None,
) -> ValidationResult:
    """
    Run the LangGraph StateGraph workflow synchronously and return the ValidationResult.
    """
    if settings is None:
        settings = Settings()
    if provider is None:
        provider = LLMProvider(settings)

    app = compile_graph(provider)

    initial_state = {
        "user_question": user_question,
        "planner_output": None,
        "retrieved_chunks": [],
        "information_output": None,
        "validation_result": None,
        "error": None,
        "retry_count": 0,
        "feedback": None
    }

    try:
        # Use asyncio.run for robust, thread-safe event loop execution
        final_state = asyncio.run(app.ainvoke(initial_state))
        val_res = final_state.get("validation_result")
        state_error = final_state.get("error")
        if state_error:
            logger.warning("[LangGraph] Pipeline completed with node-level error: {}", state_error)
        if val_res:
            return val_res
        raise ValueError("LangGraph completed execution but did not produce a validation_result.")
    except Exception as e:
        logger.exception("[LangGraph] Pipeline execution failed.")
        return ValidationResult(
            is_grounded=False,
            is_accurate=False,
            final_answer="Sorry, the LangGraph pipeline encountered an error. Please try again.",
            confidence=0.0,
            issues=[f"LangGraph pipeline error: {type(e).__name__}: {str(e)}"]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Verification block
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint
    from agents.crewai_impl.crew import run_crewai_pipeline

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

        print("\n--- Running LangGraph Pipeline ---")
        lg_res = run_langgraph_pipeline(question, settings, provider)
        
        print("\n--- Running CrewAI Pipeline ---")
        cr_res = run_crewai_pipeline(question, settings, provider)

        print("\nSide-by-Side Comparison:")
        print(f"{'Field':<20} | {'LangGraph':<35} | {'CrewAI':<35}")
        print("-" * 96)
        print(f"{'Is Accurate':<20} | {str(lg_res.is_accurate):<35} | {str(cr_res.is_accurate):<35}")
        print(f"{'Is Grounded':<20} | {str(lg_res.is_grounded):<35} | {str(cr_res.is_grounded):<35}")
        print(f"{'Confidence':<20} | {f'{lg_res.confidence:.2f}':<35} | {f'{cr_res.confidence:.2f}':<35}")
        print(f"{'Issues':<20} | {str(lg_res.issues)[:33]:<35} | {str(cr_res.issues)[:33]:<35}")
        print(f"{'Final Answer snippet':<20} | {lg_res.final_answer[:32].replace(chr(10), ' '):<35} | {cr_res.final_answer[:32].replace(chr(10), ' '):<35}")
        print()

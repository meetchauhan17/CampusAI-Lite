"""
agents/crewai_impl/tasks.py

Defines the three sequential CrewAI Tasks:
  plan_task → retrieve_task → validate_task

Context chaining ensures each task receives the previous task's output.
The expected_output strings are explicit so crewai can parse results correctly.
"""
from __future__ import annotations

from crewai import Task


def build_tasks(
    agents: dict,
    user_question: str,
    retrieved_chunks: Optional[list[dict]] = None,
) -> tuple[Task, Task, Task]:
    """
    Builds the three Tasks for a given user question.

    Args:
        agents:        Dict returned by build_agents() with keys
        'planner', 'information', 'validator'.
        user_question: The raw student query to process.
        retrieved_chunks: Optional list of pre-fetched document chunks.

    Returns:
        A tuple (plan_task, retrieve_task, validate_task) ready for Crew.
    """
    import json

    # ── Task 1: Plan ─────────────────────────────────────────────────────────
    plan_task = Task(
        description=(
            f"Analyse this student question and produce an execution plan.\n\n"
            f"Question: {user_question}\n\n"
            "Instructions:\n"
            "1. Detect the query category from: exams, fees, library, hostel, "
            "academic-calendar, or general.\n"
            "2. Break the question into 1-3 focused retrieval sub-tasks.\n"
            "3. Decide whether the university_info_search_tool is needed (requires_tool).\n\n"
            "Return ONLY a valid JSON object — no markdown, no explanation:\n"
            '{"sub_tasks": [...], "category": "...", "requires_tool": true/false}'
        ),
        expected_output=(
            'A JSON object with exactly three keys: '
            '"sub_tasks" (list of strings), '
            '"category" (string: exams/fees/library/hostel/academic-calendar/general), '
            '"requires_tool" (boolean). '
            "No markdown code fences. No prose outside the JSON."
        ),
        agent=agents["planner"],
    )

    # ── Task 2: Retrieve ─────────────────────────────────────────────────────
    chunks_str = ""
    if retrieved_chunks:
        chunks_str = f"Here are the official document chunks retrieved from the university files for this question:\n{json.dumps(retrieved_chunks, indent=2)}\n\n"

    retrieve_task = Task(
        description=(
            f"The student asked: '{user_question}'\n\n"
            "You have been given the planner's execution plan (in the context above).\n\n"
            f"{chunks_str}"
            "CRITICAL INSTRUCTIONS:\n"
            "1. You MUST call the university_info_search_tool with the student's question "
            "to retrieve relevant document chunks. (If retrieved chunks are already provided above, "
            "use them directly to answer the question, but still cite the source files correctly). "
            "DO NOT answer without using the retrieved facts. Do not rely on pre-trained knowledge or guess the answer.\n"
            "2. Based ONLY on the retrieved chunks from the tool output or provided above, compose a clear, factual answer.\n"
            "3. List every source filename you used.\n\n"
            "Provide your drafted answer and sources as a clear text response."
        ),
        expected_output=(
            "A clear, factual drafted answer based ONLY on the retrieved document chunks, "
            "including a list of source filenames used to answer."
        ),
        agent=agents["information"],
        context=[plan_task],
    )

    # ── Task 3: Validate ─────────────────────────────────────────────────────
    validate_task = Task(
        description=(
            f"Original question: '{user_question}'\n\n"
            f"{chunks_str}"
            "You have the information agent's drafted answer in the context above.\n\n"
            "STRICT VALIDATION INSTRUCTIONS:\n"
            "1. Identify every concrete factual claim in the drafted answer: "
            "dates, times, hall/block numbers, fees, amounts, deadlines, book limits.\n"
            "2. For EACH claim, compare it to the retrieved chunks above:\n"
            "   - Verify if the claim is fully supported by the retrieved chunks. A claim is supported if the chunks contain the exact same fact (e.g., date, fee amount, book limit, hall number). The wording doesn't have to be identical, but the factual information (names, dates, times, numbers) must match verbatim.\n"
            "   - If a claim is unverified (not mentioned in the chunks) or contradicts the chunks, list it as an issue in the 'issues' array. Format the issue clearly, e.g., 'Claim \"<claim>\" not found in any retrieved chunk — possible hallucination.' or 'Claim \"<claim>\" contradicts retrieved chunk: \"<chunk quote>\"'.\n"
            "3. Set is_grounded=true if all factual claims in the drafted answer are supported by the chunks.\n"
            "4. Set is_accurate=true if and only if is_grounded=true AND there are no contradictions, unverified claims, or discrepancies. If there are any discrepancies, set is_accurate=false.\n"
            "5. If the drafted answer has no issues and is completely accurate, set is_accurate=true, is_grounded=true, and issues=[].\n"
            "6. Set confidence based on verification:\n"
            "   - All claims verified and accurate: 0.90 to 1.00\n"
            "   - Most claims verified and accurate (>=75%): 0.60 to 0.89\n"
            "   - Major contradictions or unverified claims: 0.00 to 0.59\n"
            "7. In final_answer: if the drafted answer is accurate, return it as-is. If there are errors or unverified claims, correct them using ONLY the facts from the chunks.\n\n"
            "Return ONLY a valid JSON object:\n"
            '{"is_grounded": true/false, "is_accurate": true/false, '
            '"final_answer": "...", "confidence": 0.0, "issues": [...]}'
        ),
        expected_output=(
            'A JSON object with exactly five keys: '
            '"is_grounded" (bool), '
            '"is_accurate" (bool), '
            '"final_answer" (string — corrected answer using only chunk-verified facts), '
            '"confidence" (float 0.0–1.0, reflects fraction of claims verifiable in chunks), '
            '"issues" (list of strings: each unverified claim or source citation, or empty list). '
            "No markdown. No text outside the JSON."
        ),
        agent=agents["validator"],
        context=[plan_task, retrieve_task],
        output_pydantic=None,   # We parse the JSON manually in crew.py
    )

    return plan_task, retrieve_task, validate_task

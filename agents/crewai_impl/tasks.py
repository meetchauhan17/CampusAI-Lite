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
) -> tuple[Task, Task, Task]:
    """
    Builds the three Tasks for a given user question.

    Args:
        agents:        Dict returned by build_agents() with keys
                       'planner', 'information', 'validator'.
        user_question: The raw student query to process.

    Returns:
        A tuple (plan_task, retrieve_task, validate_task) ready for Crew.
    """

    # ── Task 1: Plan ─────────────────────────────────────────────────────────
    plan_task = Task(
        description=(
            f"Analyse this student question and produce an execution plan.\n\n"
            f"Question: {user_question}\n\n"
            "Instructions:\n"
            "1. Detect the query category from: exams, fees, library, hostel, "
            "academic-calendar, or general.\n"
            "2. Break the question into 1-3 focused retrieval sub-tasks.\n"
            "3. Decide whether the UniversityInfoSearchTool is needed (requires_tool).\n\n"
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
    retrieve_task = Task(
        description=(
            f"The student asked: '{user_question}'\n\n"
            "You have been given the planner's execution plan (in the context above).\n\n"
            "Instructions:\n"
            "1. Call the UniversityInfoSearchTool with the original question "
            "to retrieve relevant document chunks.\n"
            "2. Based on the retrieved chunks and the planner's sub-tasks, "
            "compose a clear, factual answer.\n"
            "3. List every source filename you used.\n\n"
            "Return ONLY a valid JSON object:\n"
            '{"raw_answer": "...", "sources": ["..."], "category": "..."}'
        ),
        expected_output=(
            'A JSON object with exactly three keys: '
            '"raw_answer" (string — the drafted answer), '
            '"sources" (list of source file names cited), '
            '"category" (string matching the planner\'s category). '
            "No markdown. No text outside the JSON."
        ),
        agent=agents["information"],
        context=[plan_task],
    )

    # ── Task 3: Validate ─────────────────────────────────────────────────────
    validate_task = Task(
        description=(
            f"Original question: '{user_question}'\n\n"
            "You have the information agent's drafted answer and sources (in context above).\n\n"
            "Instructions:\n"
            "1. Compare every fact in raw_answer against the source chunks.\n"
            "2. Set is_grounded=true only if all claims appear in the sources.\n"
            "3. Set is_accurate=true only if there are no factual errors.\n"
            "4. Correct small inaccuracies in final_answer when possible.\n"
            "5. List specific issues found (empty list if none).\n"
            "6. Assign a confidence score between 0.0 and 1.0.\n\n"
            "Return ONLY a valid JSON object:\n"
            '{"is_grounded": true/false, "is_accurate": true/false, '
            '"final_answer": "...", "confidence": 0.0, "issues": [...]}'
        ),
        expected_output=(
            'A JSON object with exactly five keys: '
            '"is_grounded" (bool), '
            '"is_accurate" (bool), '
            '"final_answer" (string — corrected answer ready for the user), '
            '"confidence" (float 0.0–1.0), '
            '"issues" (list of strings describing any problems found, or empty list). '
            "No markdown. No text outside the JSON."
        ),
        agent=agents["validator"],
        context=[plan_task, retrieve_task],
        output_pydantic=None,   # We parse the JSON manually in crew.py
    )

    return plan_task, retrieve_task, validate_task

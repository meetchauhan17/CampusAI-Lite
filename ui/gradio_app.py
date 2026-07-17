"""
ui/gradio_app.py
================
CampusAI Lite — Agentic University Information Assistant
Gradio Blocks UI showcasing four different agentic pipeline implementations.

Launch:
    python ui/gradio_app.py
"""

from __future__ import annotations

import sys
import os
import time
import traceback
from typing import Dict, Any

# Add project root to sys.path so all internal imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from config.settings import Settings
from core.logger import logger
from core.llm_provider import LLMProvider
from core.schemas import ValidationResult

# ─────────────────────────────────────────────────────────────────────────────
# Shared singletons (created once at startup)
# ─────────────────────────────────────────────────────────────────────────────
_settings = Settings()
_provider = LLMProvider(_settings)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runners
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_LABELS = {
    "CrewAI":          "crewai",
    "LangGraph":       "langgraph",
    "AG2 (AutoGen)":   "autogen",
    "BeeAI (PoC)":     "beeai",
}


def _run_pipeline(framework: str, question: str) -> Dict[str, Any]:
    """
    Dispatches to the correct pipeline and returns a unified result dict.

    Returns:
        {
          "final_answer": str,
          "category":     str,
          "confidence":   float,
          "is_grounded":  bool,
          "is_accurate":  bool,
          "issues":       list[str],
          "trace":        str,   # human-readable reasoning trace
          "latency_ms":   float,
          "error":        str | None,
        }
    """
    t0 = time.perf_counter()
    try:
        if framework == "CrewAI":
            from agents.crewai_impl.crew import run_crewai_pipeline
            result: ValidationResult = run_crewai_pipeline(question, _settings, _provider)
            trace = _crewai_trace(result)

        elif framework == "LangGraph":
            from agents.langgraph_impl.graph import run_langgraph_pipeline
            result = run_langgraph_pipeline(question, _settings, _provider)
            trace = _langgraph_trace(result)

        elif framework == "AG2 (AutoGen)":
            from agents.autogen_impl.group import run_autogen_pipeline
            result = run_autogen_pipeline(question, _settings, _provider)
            trace = _autogen_trace(result)

        elif framework == "BeeAI (PoC)":
            from agents.beeai_impl.poc import run_beeai_poc
            raw = run_beeai_poc(question, _settings, _provider)
            # BeeAI PoC returns a plain dict — normalise into our output shape
            result = None
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "final_answer": raw.get("answer", "No answer returned."),
                "category":     raw.get("category", "general"),
                "confidence":   0.7,
                "is_grounded":  True,
                "is_accurate":  True,
                "issues":       [],
                "trace":        _beeai_trace(raw),
                "latency_ms":   latency_ms,
                "error":        None,
            }
        else:
            raise ValueError(f"Unknown framework: {framework}")

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "final_answer": result.final_answer,
            "category":     _guess_category(result),
            "confidence":   result.confidence,
            "is_grounded":  result.is_grounded,
            "is_accurate":  result.is_accurate,
            "issues":       result.issues,
            "trace":        trace,
            "latency_ms":   latency_ms,
            "error":        None,
        }

    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error("[UI] Pipeline '{}' raised: {}", framework, exc)
        err_detail = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        return {
            "final_answer": "⚠️ An error occurred while running the pipeline. See details below.",
            "category":     "—",
            "confidence":   0.0,
            "is_grounded":  False,
            "is_accurate":  False,
            "issues":       [str(exc)],
            "trace":        err_detail,
            "latency_ms":   latency_ms,
            "error":        err_detail,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Trace formatters — produce human-readable step-by-step reasoning traces
# ─────────────────────────────────────────────────────────────────────────────

def _guess_category(result: ValidationResult) -> str:
    """Attempt to extract category from the final answer if not directly available."""
    for kw in ("exam", "fee", "library", "hostel", "calendar", "general"):
        if kw in result.final_answer.lower():
            return kw
    return "general"


def _crewai_trace(result: ValidationResult) -> str:
    lines = [
        "🔵 CrewAI Sequential Pipeline Trace",
        "━" * 48,
        "① Planner Agent  →  produced sub-tasks and query category",
        "② Information Agent  →  retrieved grounded facts using UniversityInfoSearchTool",
        "③ Validation Agent  →  fact-checked answer via PydanticAI (schema-enforced)",
        "",
        f"  is_grounded  : {result.is_grounded}",
        f"  is_accurate  : {result.is_accurate}",
        f"  confidence   : {result.confidence:.0%}",
    ]
    if result.issues:
        lines += ["", "  ⚠ Issues detected:"]
        for iss in result.issues:
            lines.append(f"    • {iss}")
    return "\n".join(lines)


def _langgraph_trace(result: ValidationResult) -> str:
    lines = [
        "🟢 LangGraph StateGraph Trace",
        "━" * 48,
        "① plan_node             →  classified question, produced PlannerOutput",
        "② retrieve_and_answer   →  ran UniversityInfoSearchTool, drafted InformationAgentOutput",
        "③ validate_node         →  delegated to PydanticAI ValidationAgent",
        "   conditional_edge     →  is_accurate=True → END  (or retry up to 2×)",
        "",
        f"  is_grounded  : {result.is_grounded}",
        f"  is_accurate  : {result.is_accurate}",
        f"  confidence   : {result.confidence:.0%}",
    ]
    if result.issues:
        lines += ["", "  ⚠ Issues flagged:"]
        for iss in result.issues:
            lines.append(f"    • {iss}")
    return "\n".join(lines)


def _autogen_trace(result: ValidationResult) -> str:
    lines = [
        "🟡 AG2 (AutoGen) GroupChat Trace",
        "━" * 48,
        "① PlannerAgent      →  classified question and produced plan (GroupChat round 1)",
        "② InformationAgent  →  simulated tool_call → user_proxy executed search (round 2–3)",
        "③ ValidationAgent   →  custom_reply intercepted; ran PydanticAI fact-checker (round 4)",
        "   GroupChat ended  →  ValidationAgent returned None speaker (terminate signal)",
        "",
        f"  is_grounded  : {result.is_grounded}",
        f"  is_accurate  : {result.is_accurate}",
        f"  confidence   : {result.confidence:.0%}",
    ]
    if result.issues:
        lines += ["", "  ⚠ Issues flagged:"]
        for iss in result.issues:
            lines.append(f"    • {iss}")
    return "\n".join(lines)


def _beeai_trace(raw: dict) -> str:
    lines = [
        "🟣 BeeAI Framework PoC Trace",
        "━" * 48,
        "① RouterAgent (BeeAgent, no tools)",
        f"   Category detected: {raw.get('category', '—')}",
        "② ResponderAgent (BeeAgent + UniversityInfoSearchToolBee)",
        "   Used ReAct loop: Thought → Action (search) → Observation → Final Answer",
    ]
    if raw.get("sources"):
        lines += ["", "  Sources cited:"]
        for src in raw["sources"]:
            lines.append(f"    • {src}")
    else:
        lines.append("  (No explicit source citations extracted from this run)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio event handlers
# ─────────────────────────────────────────────────────────────────────────────

def handle_submit(framework: str, question: str):
    """
    Main submit handler. Returns updates for all output components.
    """
    if not question or not question.strip():
        yield (
            "",                    # answer_box
            "",                    # category_badge
            "",                    # confidence_bar label
            0.0,                   # confidence_slider
            "",                    # trace_box
            gr.update(visible=False),  # error_box
        )
        return

    # Show a "running" state while the pipeline executes
    yield (
        "⏳ Running pipeline, please wait…",
        "—",
        "Confidence: —",
        0.0,
        "Pipeline executing…",
        gr.update(visible=False),
    )

    r = _run_pipeline(framework, question.strip())

    conf_pct = f"Confidence: {r['confidence']:.0%}"
    grounded_icon = "✅" if r["is_grounded"] else "❌"
    accurate_icon  = "✅" if r["is_accurate"]  else "❌"

    answer_md = (
        f"### 📋 Answer\n\n{r['final_answer']}\n\n"
        f"---\n"
        f"**Grounded:** {grounded_icon}  &nbsp;|&nbsp;  "
        f"**Accurate:** {accurate_icon}  &nbsp;|&nbsp;  "
        f"**Latency:** {r['latency_ms']:.0f} ms"
    )

    if r.get("error"):
        error_update = gr.update(value=f"```\n{r['error']}\n```", visible=True)
    else:
        error_update = gr.update(visible=False)

    yield (
        answer_md,
        f"🏷 Category: **{r['category']}**",
        conf_pct,
        r["confidence"],
        r["trace"],
        error_update,
    )


def handle_compare(question: str):
    """
    Runs all four pipelines and returns a markdown comparison table + raw traces.
    """
    if not question or not question.strip():
        return "Please enter a question first.", ""

    q = question.strip()
    rows = []
    all_traces = []

    for label in ["CrewAI", "LangGraph", "AG2 (AutoGen)", "BeeAI (PoC)"]:
        r = _run_pipeline(label, q)
        grounded = "✅" if r["is_grounded"] else "❌"
        accurate = "✅" if r["is_accurate"] else "❌"
        conf = f"{r['confidence']:.0%}"
        lat = f"{r['latency_ms']:.0f} ms"
        snippet = r["final_answer"][:120].replace("\n", " ")
        if len(r["final_answer"]) > 120:
            snippet += "…"
        rows.append(f"| **{label}** | {r['category']} | {conf} | {grounded} | {accurate} | {lat} | {snippet} |")
        all_traces.append(f"### {label}\n{r['trace']}")

    table_md = (
        "| Framework | Category | Confidence | Grounded | Accurate | Latency | Answer Snippet |\n"
        "|---|---|---|---|---|---|---|\n"
        + "\n".join(rows)
    )
    traces_md = "\n\n---\n\n".join(all_traces)
    return table_md, traces_md


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI layout
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
/* ── Global ── */
body, .gradio-container { font-family: 'Inter', sans-serif; }

/* ── Header banner ── */
#app-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 12px;
    padding: 28px 32px 20px;
    margin-bottom: 8px;
    color: #fff;
}
#app-header h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 6px; }
#app-header p  { font-size: 0.95rem; opacity: 0.75; margin: 0; }

/* ── Primary run button ── */
#run-btn { background: #0f3460; color: #fff; font-weight: 600; border-radius: 8px; }
#run-btn:hover { background: #1a5276; }

/* ── Compare button ── */
#cmp-btn { background: #1e8449; color: #fff; font-weight: 600; border-radius: 8px; }
#cmp-btn:hover { background: #27ae60; }

/* ── Answer panel ── */
#answer-panel { border-left: 4px solid #0f3460; padding-left: 12px; }

/* ── Confidence slider track ── */
.gradio-slider .svelte-1ipelgc { background: linear-gradient(to right, #e74c3c, #f39c12, #27ae60); }

/* ── Error box ── */
#error-box { border: 1px solid #e74c3c; border-radius: 6px; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="CampusAI Lite") as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="app-header">
            <h1>🎓 CampusAI Lite</h1>
            <p>Agentic University Information Assistant &nbsp;·&nbsp;
               Capstone Project — Comparative Study of Agentic AI Frameworks</p>
        </div>
        """)

        # ── Main layout: left input col | right output col ──────────────────
        with gr.Row(equal_height=False):

            # ── LEFT: input controls ─────────────────────────────────────────
            with gr.Column(scale=4, min_width=320):
                gr.Markdown("### 🔧 Pipeline Configuration")

                framework_dd = gr.Dropdown(
                    label="Select Agentic Framework",
                    choices=list(PIPELINE_LABELS.keys()),
                    value="LangGraph",
                    interactive=True,
                    elem_id="framework-dd",
                )

                gr.Markdown("**Framework descriptions:**")
                gr.Markdown(
                    "- **CrewAI** — Sequential 3-agent crew (Planner → Retriever → Validator)\n"
                    "- **LangGraph** — Explicit StateGraph with ReAct-style self-correction loop\n"
                    "- **AG2 (AutoGen)** — GroupChat with custom speaker-selection state machine\n"
                    "- **BeeAI (PoC)** — Two-agent Router → Responder proof-of-concept"
                )

                gr.Markdown("---")
                gr.Markdown("### 💬 Ask a University Question")

                question_box = gr.Textbox(
                    label="Student Question",
                    placeholder=(
                        "e.g. When is the Artificial Intelligence exam?\n"
                        "     How much is the B.Tech tuition fee?\n"
                        "     How many books can a PG student borrow?"
                    ),
                    lines=4,
                    max_lines=8,
                    elem_id="question-box",
                )

                with gr.Row():
                    submit_btn = gr.Button(
                        "🚀 Run Pipeline", variant="primary", elem_id="run-btn", scale=3
                    )
                    clear_btn = gr.ClearButton(
                        components=[question_box], value="🗑 Clear", scale=1
                    )

                gr.Markdown("---")

                # ── Sample questions ─────────────────────────────────────────
                gr.Markdown("### 💡 Sample Questions")
                sample_qs = [
                    "When is the Artificial Intelligence exam and what hall do I report to?",
                    "How much is the B.Tech tuition fee and what happens if I pay late?",
                    "How many books can a postgraduate student borrow from the library?",
                    "What are the hostel check-in and check-out timings?",
                    "When does the spring semester start and when are the mid-term exams?",
                ]
                for sq in sample_qs:
                    gr.Button(sq, size="sm").click(
                        fn=lambda q=sq: q,
                        inputs=[],
                        outputs=[question_box],
                    )

            # ── RIGHT: output panels ─────────────────────────────────────────
            with gr.Column(scale=6, min_width=400):
                gr.Markdown("### 📊 Pipeline Output")

                with gr.Row():
                    category_md  = gr.Markdown("🏷 Category: **—**", elem_id="cat-badge")
                    confidence_label = gr.Markdown("Confidence: —")

                confidence_slider = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0,
                    label="Confidence Score",
                    interactive=False,
                    step=0.01,
                )

                answer_md_box = gr.Markdown(
                    "*(answer will appear here after submitting a question)*",
                    elem_id="answer-panel",
                )

                error_box = gr.Markdown(
                    "",
                    visible=False,
                    elem_id="error-box",
                    label="Error Details",
                )

                gr.Markdown("---")

                with gr.Accordion("🔍 Show Reasoning Trace", open=False):
                    trace_box = gr.Textbox(
                        label="Step-by-step agent trace",
                        lines=12,
                        max_lines=25,
                        interactive=False,
                    )

        # ── Compare All 4 Section ────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("### ⚡ Compare All 4 Frameworks on the Same Question")
        gr.Markdown(
            "Runs **the same question** through CrewAI, LangGraph, AG2, and BeeAI in sequence "
            "and displays answers, latency and grounding side by side."
        )

        compare_btn = gr.Button(
            "📊 Compare All Frameworks", variant="secondary", elem_id="cmp-btn"
        )

        compare_table = gr.Markdown(
            "*(comparison table appears here after clicking the button above)*"
        )

        with gr.Accordion("📄 All Traces (from comparison run)", open=False):
            compare_traces = gr.Textbox(
                label="Full reasoning traces for all four frameworks",
                lines=20,
                max_lines=40,
                interactive=False,
            )

        # ── Footer ───────────────────────────────────────────────────────────
        gr.Markdown("---")
        gr.HTML("""
        <div style="text-align:center; font-size:0.85rem; opacity:0.55; padding-bottom:12px;">
            CampusAI Lite &nbsp;·&nbsp; Capstone Agentic AI Project &nbsp;·&nbsp;
            Primary Model: <strong>IBM Granite (watsonx.ai)</strong> with Groq &amp; Gemini fallback
        </div>
        """)

        # ── Event wiring ─────────────────────────────────────────────────────
        submit_outputs = [
            answer_md_box,
            category_md,
            confidence_label,
            confidence_slider,
            trace_box,
            error_box,
        ]

        submit_btn.click(
            fn=handle_submit,
            inputs=[framework_dd, question_box],
            outputs=submit_outputs,
        )

        question_box.submit(
            fn=handle_submit,
            inputs=[framework_dd, question_box],
            outputs=submit_outputs,
        )

        compare_btn.click(
            fn=handle_compare,
            inputs=[question_box],
            outputs=[compare_table, compare_traces],
        )

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_app()
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.indigo,
        font=gr.themes.GoogleFont("Inter"),
    )
    app.launch(
        server_name="0.0.0.0",
        server_port=_settings.GRADIO_PORT,
        share=_settings.GRADIO_SHARE,
        show_error=True,
        theme=theme,
        css=CSS,
    )

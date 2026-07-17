# Capstone Report: Agentic AI Framework Comparison

This report evaluates and compares four leading agentic AI frameworks: **CrewAI**, **LangGraph**, **AutoGen (ag2)**, and **BeeAI Framework**. The objective of this comparison is to select the most suitable framework for the **campusai-lite** university information assistant.

---

## 1. Overview of Frameworks

### CrewAI
- **Paradigm:** Role-playing and collaborative agents.
- **Key Concept:** Define Crews, Tasks, and Agents with specialized roles and tools. High level of abstraction.
- **Best Use Case:** Processes requiring a sequence of clear tasks executed by collaborating personas (e.g. researcher + writer).

### LangGraph
- **Paradigm:** Graph-based orchestration (stateful, multi-agent).
- **Key Concept:** Nodes represent actions/agent invocations, edges represent control flow. The state schema is explicitly passed and updated.
- **Best Use Case:** Highly customized, complex, cyclic agent workflows requiring fine-grained control over execution flow and state transitions.

### AutoGen (ag2)
- **Paradigm:** Conversational agent framework.
- **Key Concept:** Multi-agent conversation. Agents exchange messages to cooperatively solve tasks.
- **Best Use Case:** Dynamic, chat-based problem-solving where agents converse with other agents and humans to refine output.

### BeeAI Framework
- **Paradigm:** Agentic workflow/reasoning-loop framework.
- **Key Concept:** Built around agent capabilities, tools, and memory, focusing on robustness and production-ready enterprise integrations.
- **Best Use Case:** IBM-ecosystem aligned, structured workflows utilizing models like Granite.

---

## 2. Comparison Matrix

| Feature / Dimension | CrewAI | LangGraph | AutoGen (ag2) | BeeAI |
| :--- | :--- | :--- | :--- | :--- |
| **Abstractions** | High | Low / Medium | Medium | Medium |
| **Control Flow** | Sequential / Hierarchical | Explicit Graph (DAG/Cyclic) | Conversational / Dynamic | Iterative Reasoning Loop |
| **State Management**| Automated / Internal | Explicit State Schema | Message History | Structured Memory / Context |
| **Learning Curve** | Gentle | Steep | Moderate | Moderate |
| **Flexibility** | Moderate | Extremely High | High | Moderate |

---

## 3. Findings & Recommendation

For the **campusai-lite** platform, which acts as a university information assistant:
1. **LangGraph** offers the best control for retrieval-augmented generation (RAG) loops (e.g. self-corrective RAG) because of its explicit graph routing.
2. **CrewAI** is useful for high-level research tasks, such as comparing syllabus documents or drafting comparison reports.
3. **BeeAI** integrates natively with IBM Granite models, making it the primary choice if watsonx.ai serves as the host.

Detailed framework implementation plans will be documented in `agents/` directories.

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

---

## 4. CrewAI vs AG2 (AutoGen) Comparison

Based on the actual implementation of the three-role sequential workflow (Planner → Information → Validator) in both frameworks, we observed the following differences:

### Setup Complexity
- **CrewAI**: Setup is highly declarative and abstract. Defining agents and tasks uses straightforward class instantiations. However, using a unified local `LLMProvider` required creating a custom `CampusAICrewLLM` subclass of crewai's `BaseLLM` to route requests through our failover chain, bypassing the default LiteLLM logic.
- **AG2 (AutoGen)**: Setup is conversational. Integrating our custom LLM provider required defining a custom `CampusAIAutoGenClient` implementing AutoGen's `ModelClient` protocol, and registering it instance-by-instance via `.register_model_client()`.

### Agent-to-Agent Handoffs
- **CrewAI**: Handled implicitly by the sequential task manager. We chained tasks via the `context` parameter (`context=[plan_task, retrieve_task]`), and the framework automatically compiled inputs and directed outputs.
- **AG2 (AutoGen)**: Managed as a multi-agent chat. Enforcing a strict sequential pipeline (Planner → Retriever → Validator) required defining a custom `select_speaker_sequence` state machine that inspected the last speaker and message roles to force transitions, which bypasses AutoGen's natural conversational flow.

### Tool-Calling
- **CrewAI**: Straightflow integration. We defined a CrewAI-native `BaseTool` and passed it directly to the agent's tool list.
- **AG2 (AutoGen)**: Wired via `register_function` pairing a caller (InformationAgent) and an executor (user_proxy). Since our custom model client wrapped a text-only `LLMProvider`, we had to simulate structured `tool_calls` inside the custom client's `create()` method so the proxy could execute the search, adding orchestration overhead.

### Verbosity & Debuggability
- **CrewAI**: High verbosity with colourful logs detailing agent thoughts, actions, and observations. Extremely readable for developers but verbose for production.
- **AG2 (AutoGen)**: Logs are printed as chat messages (`PlannerAgent (to chat_manager): ...`). Good for conversational flows, but tracing tool-call routing and custom client errors requires looking deep into the console printouts.

### Summary: Suitability for Sequential Pipelines
- **CrewAI** is a more natural fit for strict sequential pipelines. Its native Task chaining abstraction makes it easy to write clean, predictable workflows.
- **AG2 (AutoGen)** is designed for open-ended conversational problem solving. Forcing it into a strict pipeline requires writing custom transition functions and registering custom replies, making it feel less suited for rigid processes but highly powerful for interactive feedback loops.


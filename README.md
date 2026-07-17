# 🎓 CampusAI-Lite: Multi-Framework University Assistant

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Frameworks](https://img.shields.io/badge/Frameworks-CrewAI%20%7C%20LangGraph%20%7C%20AutoGen%20%7C%20BeeAI-orange.svg)](#)
[![Backend](https://img.shields.io/badge/Backend-FastAPI%20%7C%20Uvicorn-green.svg)](#)
[![Frontend](https://img.shields.io/badge/Frontend-Gradio%20%7C%20Next.js-blueviolet.svg)](#)

CampusAI-Lite is a high-performance, Capstone-level **Agentic University Information Assistant** designed to compare and evaluate multiple agentic AI frameworks—**CrewAI**, **LangGraph**, **AutoGen (AG2)**, and **BeeAI Framework**—using structured evaluation workloads.

It answers complex university inquiries (fees, hostel rules, exam timetables, etc.) by retrieving grounded context from academic documents and validating answers for accuracy.

---

## 🏗️ Core Architecture & Data Flow

CampusAI-Lite is structured around a decoupled agent-and-server architecture:

```
                  ┌─────────────────────────────────────────┐
                  │            Next.js / Gradio UIs         │
                  └────────────────────┬────────────────────┘
                                       │ POST /api/compare
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │         FastAPI Comparison Engine       │
                  │   (ThreadPoolExecutor Parallel Runner)  │
                  └────┬──────────┬──────────┬──────────┬───┘
                       │          │          │          │
                       ▼          ▼          ▼          ▼
                    [CrewAI] [LangGraph]  [AutoGen]  [BeeAI]
                       │          │          │          │
                       └──────────┼──────────┼──────────┘
                                  ▼
                   ┌──────────────────────────────────────┐
                   │          Unified LLMProvider         │
                   │ (Thread Lock & Model Normalization)  │
                   └──────────────┬───────────────────────┘
                                  │
                                  ├─► Watsonx.ai ("bob" CLI) [Primary]
                                  ├─► Groq Cloud             [Fallback 1]
                                  └─► Google Gemini          [Fallback 2]
```

### 1. Unified `LLMProvider`
All framework pipelines run their inference calls through a single [LLMProvider](file:///c:/Meet/xyz/campusai-lite/core/llm_provider.py) class. This guarantees identical model characteristics and reliable fallbacks:
* **Failover Chain**: `bob` (watsonx.ai CLI wrapper) $\rightarrow$ `groq` $\rightarrow$ `gemini`.
* **Model Normalization**: Unsupported Granite model tags (e.g. `ibm/granite-3-8b-instruct`) are mapped to `gemini-3.5-flash` to keep the Bob CLI stable.
* **Concurrency Lock**: A global thread lock prevents Git config locking conflicts (`.git/config: File exists`) when parallel frameworks call the Bob CLI simultaneously.

### 2. Retrieval & Ingestion Pipeline
* **Vector Store**: Powered by ChromaDB.
* **Embeddings**: Uses `all-MiniLM-L6-v2` via `SentenceTransformer` to embed documents into 384-dimensional vectors.
* **Ingestion**: Scans [data/university_docs](file:///c:/Meet/xyz/campusai-lite/data/university_docs) and chunks text into categories (`academic-calendar`, `exams`, `fees`, `hostel`, `library`).

### 3. FastAPI Parallel Runner
The `/api/compare` endpoint executes all 4 framework pipelines concurrently using a `ThreadPoolExecutor`. This reduces comparison API latency from ~40s down to **~10–12s** (the runtime of the slowest single framework).

---

## 📁 Folder Structure

```
campusai-lite/
├── .env.example             # Environment variables template
├── .gitignore               # Excludes virtual environments, .env, and logs
├── requirements.txt         # Pinned Python package dependencies
├── README.md                # This setup and architecture guide
├── test_bob_api.py          # (Temporary/Removed) Bob integration script
├── config/
│   └── settings.py          # Config loader and validator via Pydantic
├── core/
│   ├── llm_provider.py      # Multi-provider client with retry and thread locks
│   ├── schemas.py           # Unified request/response validation schemas
│   ├── logger.py            # Console reconfiguration to UTF-8 on Windows
│   └── ingestion.py         # Vector DB manager (ChromaDB + SentenceTransformers)
├── tools/
│   └── university_search_tool.py # Search tool with CrewAI, LangChain wrappers
├── data/
│   └── university_docs/     # Academic guidelines, fee schedules, timetables
├── agents/
│   ├── crewai_impl/         # CrewAI Sequential Workflow (Planner -> Info -> Validator)
│   ├── langgraph_impl/      # LangGraph cyclical statechart implementation
│   ├── autogen_impl/        # AutoGen group chat agent implementation
│   └── beeai_impl/          # BeeAI PoC framework implementation
├── ui/
│   ├── gradio_app.py        # Gradio interface for quick playground tests
│   └── web/                 # Next.js web application comparison dashboard
├── reports/
│   └── framework_comparison.md # Detailed evaluation of agent frameworks
└── tests/                   # Test suite (LLM, vector search, validation)
```

---

## 🛠️ Setup & Installation

### Prerequisite
Python **3.11** must be installed on your system.

### 1. Clone & Prepare Virtual Environment
Create and activate a virtual environment in the project root:

```powershell
# Create venv
py -3.11 -m venv .venv

# Activate (PowerShell)
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Fill in your API credentials:
```env
PRIMARY_PROVIDER=bob
BOB_MODEL=ibm/granite-3-8b-instruct
BOBSHELL_API_KEY=your_bob_key
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

---

## 🚀 Running the Services

Always ensure your virtual environment is active (`.venv\Scripts\Activate.ps1`) before executing.

### 1. Ingest Documents (Initialize Vector Store)
Ingest the university rules into ChromaDB:
```bash
python -m core.ingestion
```

### 2. Start the Backend API Server
Start the FastAPI server on port `8000`:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
* Interactive API Documentation will be available at `http://localhost:8000/docs`.

### 3. Launch the Next.js Web App
Navigate to the web UI directory, install Node modules, and run the Next.js development server on port `3000`:
```bash
cd ui/web
npm install
npm run dev -- -p 3000
```
* Open your browser to `http://localhost:3000` to access the comparison dashboard.

### 4. Launch the Gradio Playgroud UI
Launch the interactive python-based playground:
```bash
python ui/gradio_app.py
```
* Open your browser to `http://localhost:7860`.

---

## 🧪 Testing & Verification

Run the full suite of unit tests to verify the tool classifications, LLM provider, and PydanticAI validation routines:

```bash
python -m unittest discover -s tests
```

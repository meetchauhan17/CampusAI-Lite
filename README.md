# CampusAI-Lite: University Info Assistant

CampusAI-Lite is a prototype **Agentic University Information Assistant** built as a Capstone project. The project is designed to compare and evaluate multiple agentic AI frameworks: **CrewAI**, **LangGraph**, **AutoGen (ag2)**, and **BeeAI Framework** using standard and structured evaluation workloads.

Under the hood, our primary provider is designated as **"bob"** which interfaces with the **IBM watsonx.ai** SDK (with fallback options to Groq and Google Gemini).

---

## Folder Structure

```
campusai-lite/
├── .env.example            # Environment variables template
├── .gitignore              # Standard git exclusion configurations
├── requirements.txt        # Pinned Python package dependencies
├── README.md               # Setup and project guide
├── config/
│   └── settings.py         # Config loader & validator (Pydantic BaseSettings)
├── core/
│   ├── llm_provider.py     # LLM integration client (watsonx.ai "bob", Groq, Gemini)
│   ├── schemas.py          # Unified request/response validation schemas
│   └── logger.py           # Structured logger configuration (Loguru)
├── tools/
│   └── __init__.py         # Custom utility tools package
├── data/
│   └── university_docs/    # Sample PDF syllabus, guidelines, and manuals
├── agents/
│   ├── crewai_impl/        # CrewAI implementation workflow
│   ├── langgraph_impl/     # LangGraph statechart implementation
│   ├── autogen_impl/       # AutoGen conversation-based implementation
│   └── beeai_impl/         # BeeAI developer framework workflow
├── ui/
│   ├── gradio_app.py       # Gradio UI for quick interactive tests
│   └── web/                # Placeholder directory for a Next.js UI frontend
├── reports/
│   └── framework_comparison.md  # Detailed framework evaluation report
└── tests/                  # Package unit tests
```

---

## Setup & Installation

Follow these steps to set up the project locally:

### 1. Clone the Repository
Clone the project into your local workspace.

### 2. Create the Virtual Environment
Create a Python 3.11 virtual environment in the root directory:
```bash
py -3.11 -m venv .venv
```

### 3. Activate and Install Dependencies
Activate the virtual environment and install the pinned dependencies:

**On Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

**On Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**On macOS/Linux:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
Open `.env` and fill in your API credentials:
- **BOBSHELL_API_KEY** and **BOB_PROJECT_ID** come from your IBM Cloud watsonx.ai platform.
- **GROQ_API_KEY** comes from the [Groq Console](https://console.groq.com).
- **GEMINI_API_KEY** comes from [Google AI Studio](https://aistudio.google.com).

### 5. Run the Gradio UI
Launch the interactive Gradio interface:
```bash
python ui/gradio_app.py
```
Open your browser and navigate to `http://127.0.0.1:7860`.

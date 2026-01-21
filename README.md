# Agentic Reasoning Engine

[![Python Version](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An autonomous, self-correcting RAG system that leverages **Graph-based State Machines** to perform deep reasoning, multi-step research, and hallucination monitoring.

Unlike traditional linear RAG, this engine operates as a cognitive agent: it verifies its own answers, refines search queries when results are poor, and intelligently routes questions between fast internal knowledge and deep web research.

## Architectural Overview

The core logic is built on **LangGraph**, orchestrating a cyclic state machine with the following capabilities:

1.  **Smart Routing & Intent Detection**:
    - Uses **Gemini 2.5 Pro** as a router to classify intent (General Chat vs. Technical Research).
    - Dynamically switches between the **Vector Store** (Qdrant) for internal knowledge and **Web Search** (DuckDuckGo) for real-time data.

2.  **Self-Correction Loops**:
    - **Grader Node**: Evaluates document relevance before generation. If retrieved docs are irrelevant, it triggers a **Query Refinement** step to rewrite the search terms.
    - **Hallucination Monitor**: Checks the final answer against facts. If ungrounded, it rejects the answer and forces a retry.

3.  **Observability & Analytics**:
    - Integrated with **Langfuse** for end-to-end tracing.
    - Tracks token usage, latency per node, and full execution paths for debugging complex agent behaviors.

## Directory Structure

```bash
├── frontend/             # React + Vite + Tailwind (Pro Max UI)
│   ├── src/              # Frontend components & logic
│   └── tailwind.config.js
├── src/                  # Core Agent Logic
│   ├── graph/            # LangGraph State Machine
│   │   ├── nodes/        # Individual Agent Steps (Search, Grade, Generate)
│   │   ├── state.py      # Shared Agent State Schema
│   │   └── workflow.py   # Graph Topology Compilation
│   ├── main.py           # FastAPI Entrypoint
│   └── vectorstore.py    # Qdrant Integration
├── docker-compose.yml    # Vector Database Infrastructure
└── requirements.txt      # Python Dependencies
```

## Installation

Requires **Python 3.12+**, **Node.js 20+**, and **Docker**.

```bash
git clone https://github.com/AbeneilMagpantay/Agentic-Reasoning-Engine.git
cd Agentic-Reasoning-Engine

# 1. Install Python Backend Dependencies
pip install -r requirements.txt

# 2. Install Frontend Dependencies
cd frontend
npm install
cd ..
```

## Configuration

Create a `.env` file in the root directory:

```ini
# Core AI
GOOGLE_API_KEY=your_gemini_key

# Vector Database
QDRANT_URL=http://localhost:6333

# Observability (Optional)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Usage

### 1. Infrastructure (Vector DB)
Start the Qdrant instance using Docker:
```bash
docker-compose up -d
```

### 2. Backend API
Launch the FastAPI server (Agent Logic):
```bash
uvicorn src.main:app --port 8000 --reload
```

### 3. Frontend UI
Launch the React Interface:
```bash
cd frontend
npm run dev
```

## Disclaimer

This project is an experimental implementation of Agentic RAG patterns. While effective, AI models can still hallucinate. Always verify important information.

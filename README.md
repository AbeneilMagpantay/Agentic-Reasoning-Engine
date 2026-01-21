# Agentic Reasoning Engine

> "An autonomous, self-correcting knowledge system that doesn't just retrieve—it reasons."

## Overview
The **Agentic Reasoning Engine** is a production-grade RAG platform designed for high-stakes domains. Unlike traditional RAG systems that simply retrieve and generate, this engine actively verifies its own answers, refines queries when necessary, and monitors for hallucinations.

## Key Features
*   **Dual-Path Routing**:
    *   **Fast Path**: Handles chitchat and greetings instantly.
    *   **Agentic Path**: Triggers deep reasoning for complex queries.
*   **Self-Correction**: Automatically detects hallucinations and "not useful" answers, triggering regeneration.
*   **Query Refinement**: If retrieval fails, the agent rewrites the query to find better documents.
*   **Tech Stack**: LangGraph, Gemini 2.5 Flash, Qdrant, FastAPI, React + Tailwind (Frontend).

## Architecture
The system implements a **Smart Routing** mechanism that balances speed and reliability.

### Workflow Diagram

```mermaid
graph TD
    UserInput[User Input] --> Router{Router Node<br/>(Gemini 2.5)}
    
    %% Path 1: Fast Mode (Chitchat)
    Router -- "General / Chitchat" --> GeneratorFast[Generator<br/>(Creative Prompt)]
    GeneratorFast --> End([Response])
    
    %% Path 2: Agentic Mode (Reasoning)
    Router -- "Technical / Complex" --> Retrieve[Retriever<br/>(Qdrant Vector DB)]
    Retrieve --> Grade{Grader Node<br/>(Rate Documents)}
    
    Grade -- "Irrelevant Docs" --> Decision1{Retry Limit?}
    Decision1 -- "Under Limit" --> Refine[Query Refiner<br/>(Rewrite Query)]
    Refine --> Retrieve
    Decision1 -- "Max Retries" --> GeneratorFast
    
    Grade -- "Relevant Docs" --> Generate[Generator<br/>(RAG Context)]
    Generate --> Hallucination{Hallucination Monitor<br/>(Check Facts)}
    
    Hallucination -- "Hallucinated / Not Useful" --> Decision2{Retry Limit?}
    Decision2 -- "Under Limit" --> Generate
    Decision2 -- "Max Retries" --> End
    
    Hallucination -- "Grounded & Useful" --> End
```

## Getting Started

### Prerequisites
*   Docker (for Qdrant)
*   Python 3.12+
*   Node.js & npm (for Frontend)
*   Google Cloud API Key (Gemini)

### Installation

1.  **Clone & Setup**:
    ```bash
    git clone <repo-url>
    cd Agentic-Reasoning-Engine
    ```

2.  **Environment Variables**:
    Create `.env`:
    ```ini
    GOOGLE_API_KEY=your_key_here
    QDRANT_URL=http://localhost:6333
    ```

3.  **Start Services**:
    ```bash
    # Start Vector DB
    docker-compose up -d
    
    # Ingest Sample Data
    python -m src.ingest
    ```

4.  **Run Backend**:
    ```bash
    uvicorn src.main:app --reload --port 8000
    ```

5.  **Run Frontend**:
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

## Design System (Frontend)
Based on **UI/UX Pro Max** principles:
*   **Style**: AI-Native, Dark Mode, Bentogrids.
*   **Palette**: Deep Tech (Zinc/Slate base, Purple accents).
*   **Icons**: Lucide React.

# Agentic Reasoning Engine: Project Specification & Implementation Guide

## Project Identity
*   **Project Name:** Agentic Reasoning Engine
*   **Tagline:** "An autonomous, self-correcting knowledge system that doesn't just retrieve—it reasons."
*   **Vision:** Build and deploy a production-grade RAG platform designed for high-stakes domains (Legal, Cyber, Fintech) where hallucinations are unacceptable.

## Core Philosophy
Most RAG systems follow a naive pattern: Retrieve → Generate. This creates a fundamental reliability problem. The Agentic Reasoning Engine inverts this:
*   **Treats retrieval as a tool**, not a pipeline.
*   **Verifies its own answers** before surfacing them.
*   **Loops back to correct mistakes** autonomously.
*   **Core Principle:** Optimize for reliability, not latency.

---

## Tech Stack (Production-Grade)
| Component | Choice | Rationale |
| :--- | :--- | :--- |
| **Orchestration** | `LangGraph` | Stateful, cyclic multi-agent flows; production-proven. |
| **LLM Provider** | `Google Gemini 2.5 Pro` | Superior reasoning for agents, free tier. |
| **Vector DB** | `Qdrant` | Self-hosted via Docker; superior filtering. |
| **Observability** | `Langfuse` | Traces every step, cost tracking, CI/CD integration. |
| **Eval Framework** | `Ragas` | Automated scoring for Faithfulness, Context Precision. |
| **API** | `FastAPI` | Async-first, streaming support (SSE). |

---

## The Agentic Architecture (StateGraph Implementation)

### Node 1: Router / Planner
*   **Purpose:** Analyze query, decide strategy (Vector vs Web vs Logic).
*   **Output:** Structured JSON (`query_type`, `retrieval_needed`, `strategy`).

### Node 2: Retriever
*   **Purpose:** Fetch documents using semantic + lexical search.
*   **Technique:** HyDE (Hypothetical Document Embeddings) for better semantic matching.
*   **Process:** Hybrid Search (Vector + BM25) -> Reranking (Cohere/Cross-Encoder).

### Node 3: Grader (The Critic)
*   **Purpose:** Validate retrieved documents are relevant.
*   **Decision Rule:** Puts documents into "Pass" or "Fail" buckets.
*   **Transition:** If too many fail -> **Trigger Query Rewrite** (Node 3a) -> Loop back to Retriever.

### Node 4: Generator
*   **Purpose:** Synthesize answer using *only* graded documents.
*   **Constraint:** Must cite sources `[doc_1]`. No external knowledge injection.

### Node 5: Hallucination Monitor (The Judge)
*   **Purpose:** Verify answer matches documents (Faithfulness) and Query (Completeness).
*   **Decision:**
    *   **Pass:** -> Stream to user.
    *   **Fail:** -> **Retry Generation** (Self-Correction loop).

### Node 6: Output Formatter (Streaming)
*   **Purpose:** Stream real-time "thought process" via SSE (Server-Sent Events) to build user trust.

---

## "Senior" Features to Implement
1.  **Structured Outputs:** Use Pydantic models for every node decision (no parsing errors).
2.  **Async Streaming:** Show the "Agent's Brain" working in real-time.
3.  **Evaluation Pipeline (CI/CD):**
    *   GitHub Actions runs `Ragas` on every push.
    *   Fails build if `Context Precision < 0.80` or `Faithfulness < 0.85`.
4.  **Observability:** Full Langfuse integration to trace costs and latency.

---

## Implementation Roadmap
1.  **Phase 1 (Weeks 1-2):** Core StateGraph (Router -> Retriever -> Grader -> Generator).
2.  **Phase 2 (Weeks 3-4):** Agentic Loops (Query Rewriter, Hallucination Monitor).
3.  **Phase 3 (Weeks 5-6):** Observability & Streaming (Langfuse).
4.  **Phase 4 (Weeks 7+):** Golden Test Sets & CI/CD Pipelines.

# SHL Assessment Recommender: Technical Approach

## 1. Architecture Overview
The system is built as a **Retrieval-Augmented Generation (RAG)** agent. It bridges the gap between vague recruiter intent and specific SHL assessments by combining semantic vector search with a high-reasoning LLM (Llama-3.3-70b via Groq).

### Stack Decisions:
- **FastAPI**: Chosen for its high performance and native Pydantic support, ensuring 100% schema compliance for automated evaluators.
- **ChromaDB**: Utilized as the vector store with the `all-MiniLM-L6-v2` embedding model for fast, local semantic retrieval.
- **Groq (Llama-3.3-70b)**: Selected for its sub-second latency and superior reasoning capabilities required for complex behaviors like comparison and refusal.

## 2. Retrieval Strategy: Multi-Pass Merged Search
Initially, handling "Refinement" was a challenge as single-pass search on the full history often "drowned out" new keywords. We evolved this into a **Multi-Pass Merged Retrieval** logic:
1. **Latest Intent Pass**: Searches specifically on the latest user message to capture immediate pivots.
2. **Structured Signal Pass**: Extracts key signals (role, level, skills, purpose) from the entire conversation history to build a high-precision query (e.g., `role: engineer level: senior skills: java aws`).
3. **Context Pass**: Searches based on the last 3 turns to maintain broader continuity.
4. **Merge & Deduplicate**: All results are merged and deduplicated by URL, ensuring a rich 25-item context window that respects both long-term goals and short-term changes.

## 3. Prompt Engineering & Behavior Design
The system prompt is fine-tuned to balance accuracy with helpfulness:
- **Smart Clarification**: The agent is instructed to skip clarification if a job title or specific skill is already mentioned, ensuring a faster path to recommendations.
- **Fast-Track Rule**: If a user message is >30 words or explicitly contains "job description", the agent is prompted via a system hint to bypass further clarification and recommend immediately.
- **Richer Shortlists**: The agent now prefers recommending a broader set of relevant assessments (up to 10) instead of being overly conservative.
- **Grounding & Verbatimness**: Verbatim URL checks and strict grounding in the provided context remain core constraints.

## 4. Evaluation Approach
We utilize an automated `eval.py` suite:
- **Recall@10**: Our multi-pass retrieval significantly improved Mean Recall@10 from **0.54** to **0.74**.
- **Behavior Probes**: 100% pass rate on probes for off-topic refusal, vague intent clarification, and prompt injection defense.

## 5. Lessons Learned
- **Latency vs. Context**: We initially tried a second LLM call to refine recommendations but found it caused timeouts. Moving all retrieval logic *before* a single, high-context LLM call proved more robust.
- **JSON Robustness**: Pre-escaping curly braces in the system prompt and using regex for extraction ensures the API never breaks even when the LLM adds conversational filler.

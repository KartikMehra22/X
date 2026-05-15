# SHL Assessment Recommender: Technical Approach

## 1. Architecture Overview
The system is built as a **Retrieval-Augmented Generation (RAG)** agent. It bridges the gap between vague recruiter intent and specific SHL assessments by combining semantic vector search with a high-reasoning LLM (Llama-3.3-70b via Groq).

### Stack Decisions:
- **FastAPI**: Chosen for its high performance and native Pydantic support, ensuring 100% schema compliance for automated evaluators.
- **ChromaDB**: Utilized as the vector store with the `all-MiniLM-L6-v2` embedding model for fast, local semantic retrieval.
- **Groq (Llama-3.3-70b)**: Selected for its sub-second latency and superior reasoning capabilities required for complex behaviors like comparison and refusal.

## 2. Retrieval Strategy: Dual-Pass Search
One of the key challenges was handling "Refinement" (e.g., adding personality tests to an existing tech shortlist). A single-pass search on the whole history often "drowned out" new, semantically distant keywords like "personality" with the original "Java developer" context.

**Our Solution**: We implemented a **Dual-Pass Retrieval** logic:
1. **Context Pass**: Searches based on the last 3 turns to maintain continuity.
2. **Intent Pass**: Searches based specifically on the latest user message to capture immediate pivots.
3. **Merge & Deduplicate**: The results are merged, ensuring that if a user says "actually, add personality tests," those items appear in the top 25 context window even if they aren't "Java" related.

## 3. Prompt Engineering & Behavior Design
The system prompt is structured to enforce the **CLARIFY -> RECOMMEND -> REFINE** pipeline.
- **Grounding**: The agent is explicitly forbidden from using prior knowledge for comparisons. It must use the `### CONTEXT` provided in the prompt.
- **URL Verbatimness**: To prevent hallucinations, the LLM is instructed to treat URLs as immutable strings from the context.
- **Scope Control**: Specific blocks in the prompt handle "Refusal" behaviors for legal advice (e.g., NYC Law 144) and general hiring tips.

## 4. Evaluation Approach
We developed a custom `eval.py` suite to measure performance against the 10 provided conversation traces.
- **Schema Validation**: Automated check for JSON structure and verbatim URLs.
- **Recall@10**: Measured the overlap between agent recommendations and the labeled "expected shortlist" in traces.
- **Behavior Probes**: Scripted tests for refusal of off-topic queries and turn-cap (8 turns) compliance.

## 5. Lessons Learned
- **Initial Failure**: Early versions used Python's `.format()` on the system prompt, which crashed when the LLM returned JSON containing curly braces. We solved this by pre-escaping prompt braces and using robust regex-based JSON extraction.
- **Vibe-Coding vs. Engineering**: We moved away from simple "keyword matching" to a semantic vector-first approach to better handle synonyms (e.g., "coding challenge" matching "Automata simulation").

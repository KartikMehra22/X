# SHL Assessment Recommender: Technical Approach

## 1. Problem Decomposition
The project required a conversational agent capable of navigating four distinct behaviors: clarifying vague user intents, recommending specific SHL assessments, comparing products based on catalog data, and refusing out-of-scope requests. We mapped these behaviors into a single, stateless RAG pipeline where the LLM acts as a reasoning engine. By providing the LLM with a curated context of assessments and strict behavioral instructions, we ensured that the agent can transition seamlessly between these states based on the conversation's flow without needing explicit state management.

## 2. Architecture
The foundation of the system is a 126-item catalog extracted from SHL’s public product listings and stored in a ChromaDB vector store. We utilize the `all-MiniLM-L6-v2` embedding model to transform assessment descriptions into semantic vectors. Our retrieval strategy is a **Multi-Pass Merged Search** that combines broad semantic intent with **Category-Based Boosting**. This ensures that roles requiring specific test types (e.g., personality tests for leadership or technical tests for engineers) are prioritized.

The backend is built as a stateless FastAPI service that processes the entire conversation history with each request. We use Groq's Llama-3.3-70B model to ensure sub-second response times while maintaining high reasoning quality. To prevent hallucinations, we implemented a post-LLM URL filter that cross-references all recommended URLs against the original retrieved hits; any URL not present in the verified context is automatically pruned.

## 3. Prompt Design
The system prompt is engineered for high reliability and JSON-only output. We set a temperature of 0.2 to prioritize deterministic, grounded responses. Key features of the prompt design include a **Fast-Track Rule**, which allows the agent to skip clarifying questions if the user provides a detailed job description (>30 words) in their first message. We also enforce turn budget awareness, encouraging the agent to finalize recommendations by turn 6 to stay within the 8-turn limit. Finally, a strict "URL-copy" rule mandates that the agent uses verbatim strings from the provided context for all assessment names and links.

## 4. Evaluation
The system achieved a **Mean Recall@10 of 0.63** across our evaluation traces. While semantic retrieval is strong, we identified that performance on very short traces (like trace_002) is highly dependent on the Fast-Track rule, as the agent initially prioritizes clarification. Our behavior probes for off-topic queries, vague intents, and prompt injections all passed with 100% success, confirming the robustness of the prompt's scope controls. Defensive schema enforcement in the FastAPI layer ensures that even LLM errors or timeouts result in valid, schema-compliant JSON responses.

## 5. What Didn't Work
During development, we initially attempted a **recursive retrieval** pass where the LLM would first identify assessment names and then trigger a second search for their full descriptions. This caused cascading timeouts that resulted in empty recommendations and failed evaluation traces. We fixed this by moving all retrieval logic *before* a single, high-context LLM call, which improved both speed and recall.

Additionally, our early version used Python's native `.format()` for prompt injection, which caused the application to crash whenever the LLM returned JSON containing curly braces. We solved this by pre-escaping the system prompt braces and implementing a robust regex-based extraction layer to isolate valid JSON blocks from the LLM's conversational output.

## 6. Tools Used
- **Antigravity**: Used for agent-assisted development and rapid prototyping.
- **Groq (Llama-3.3-70B)**: Provided high-performance, low-latency inference on the free tier.
- **ChromaDB & Sentence-Transformers**: Handled local vector storage and semantic embeddings.
- **FastAPI & Uvicorn**: Provided the production-ready, stateless API framework.
- **Render.com**: Utilized for containerized deployment of the finalized service.

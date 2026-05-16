# SHL AI Intern Assignment - Technical Approach

## 1. Problem Decomposition
The goal was to build a bot that helps recruiters pick the right SHL tests. I broke this down into four main jobs: clarifying when the user is too vague, recommending assessments when there's enough context, comparing different tests, and refusing out-of-scope stuff (like legal advice). I decided to map all of this to a single RAG pipeline where the LLM gets the conversation history and a bunch of relevant catalog chunks, then decides which behavior to trigger based on the latest message.

## 2. Architecture
I started by scraping about 126 assessments into a JSON file, which I then indexed in ChromaDB using the MiniLM-L6-v2 model. I went with a multi-pass retrieval approach: one pass for the latest user intent and another structured pass to catch long-term signals like roles or seniority levels. I also added a "category boost" to make sure personality tests actually show up for manager roles, as pure semantic search sometimes missed them.

The backend is a stateless FastAPI app deployed at **https://x-t52m.onrender.com**. I used Groq (Llama-3 70B) because the latency was much better than the Gemini free tier in my early tests. To stop the LLM from making up URLs, I added a filter at the end that only allows links that were actually found in the vector store hits for that turn.

## 3. Prompt Design
I kept the prompt focused on getting valid JSON back. I found that setting the temperature to 0.2 made the recommendations much more consistent. I also added a "Fast-Track" rule: if the user pastes a whole job description, the bot skips the small talk and goes straight to recommendations. To stay under the turn limit, I told the LLM to wrap things up by turn 6 or 7.

## 4. Evaluation
My Mean Recall@10 is currently 0.63 across the 5 public traces. Some of the lower scores come from cases where the bot wanted more clarification before recommending, which I think is actually better for a real user even if it hurts the metric slightly. I also ran three behavior probes (off-topic, vague, and prompt injection) and they all passed.

## 5. What Didn't Work
My first attempt at retrieval was too simple—just taking the last message and looking for matches. This failed for management roles because "manager" is a broad term and semantic search would often return generic skill tests instead of personality assessments. I fixed this by adding the "type-based boosting" logic.

Another thing that bit me was trying to do a "second pass" LLM call to refine the list. It worked locally, but once I started testing with the evaluator's 30s timeout, it would occasionally time out and return nothing. I eventually realized it was better to just give the LLM more context upfront in one single call and let it handle the filtering.

## 6. Tools Used
I used Antigravity to help with the heavy lifting of the initial implementation. Groq's free tier was great for the LLM logic, and ChromaDB handled the vector storage. I also used sentence-transformers for the local embeddings and Render.com for the final deployment.

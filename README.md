# SHL Assessment Recommender

A conversational agent that helps hiring managers find SHL assessments. Built for the SHL Labs AI intern take-home.

The bot uses a RAG setup with ChromaDB to search through a catalog of ~126 SHL assessments. It's powered by Groq (Llama 3 70B) to keep the responses fast and accurate.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your Groq API key to a `.env` file:
```env
GROQ_API_KEY=your_key_here
```

3. (Optional) Run the scraper and indexer if you want to refresh the data:
```bash
python scraper.py
python build_index.py
```

4. Start the app:
```bash
python main.py
```

The server runs on `http://localhost:8000`.

## API Usage

### POST /chat
The main endpoint. Expects a list of messages.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need to hire a senior java dev"}]}'
```

### GET /health
Just a simple health check.
```bash
curl http://localhost:8000/health
```

## Notes

- The scraper is a bit fragile — if SHL changes their HTML structure it'll need updating.
- Recall@10 is 0.63 on the public traces. I noticed that the bot sometimes prefers to clarify before recommending, which can lower the score but feels more natural.
- Cold start on Render takes about 90 seconds because the embedding model needs to load into memory.
- I used Groq instead of Gemini mainly because the latency was better (sub-second in most cases).

## Deployment

Deployed on Render using the Dockerfile. It's set up to listen on port 8000.

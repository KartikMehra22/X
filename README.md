# SHL Assessment Recommender

An AI-powered advisor that helps HR professionals and hiring managers find the perfect SHL assessments through dialogue. Using a RAG (Retrieval-Augmented Generation) architecture, the system semantically searches a comprehensive catalog of 120+ SHL assessments and utilizes the Groq Llama-3.3-70b model to provide tailored, fact-based recommendations. The agent supports complex conversational behaviors including clarification, refinement, comparison, and refusal of out-of-scope queries.

## Setup Instructions

### 1. Installation
Ensure you have Python 3.11+ installed.
```bash
pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Data Ingestion & Indexing
Run the scraper to collect the latest catalog data and then build the vector index:
```bash
python scraper.py
python build_index.py
```

### 4. Start the Server
```bash
python main.py
```
The server will be available at `http://localhost:8000`.

## API Examples

### Health Check
**Request:**
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{"status": "ok"}
```

### Chat & Recommendation
**Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need a personality test for a senior marketing manager role."}]}'
```
**Response:**
```json
{
  "reply": "Based on the seniority and role, I recommend the Occupational Personality Questionnaire (OPQ)...",
  "recommendations": [
    {
      "name": "OPQ32r",
      "url": "https://www.shl.com/products/...",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

## Deployment
This project is configured for deployment on **Render.com** using the included `Dockerfile` and `render.yaml`.

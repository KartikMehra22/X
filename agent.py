import os
import json
import re
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global ChromaDB Setup
CHROMA_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_or_create_collection(name="shl_catalog", embedding_function=embedding_func)

# Groq Client Setup
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)

SYSTEM_PROMPT = """You are an expert SHL Assessment Advisor. Your goal is to guide recruiters and hiring managers from a vague intent to a grounded shortlist of SHL assessments.

### CONTEXT FROM CATALOG:
{catalog_context}

### BEHAVIORS:
1. CLARIFY: If the user's request is vague (e.g., "I need an assessment"), ask 1-3 targeted clarifying questions about job level, specific skills, or assessment type (simulation vs. knowledge-based).
2. RECOMMEND: Once you have enough context, recommend 1-10 assessments. For each, provide the EXACT name and URL from the context.
3. REFINE: If the user changes constraints mid-conversation, update the recommendations based on the new criteria while maintaining the context of the previous conversation.
4. COMPARE: If the user asks to compare specific assessments (e.g., "What is the difference between OPQ and GSA?"), provide a grounded comparison based ONLY on the descriptions and metadata in the provided context. Do not use outside knowledge.
5. REFUSE: Strictly refuse to provide general hiring advice, legal advice, or answer non-SHL related questions. Stay within the scope of SHL product recommendations.

### OUTPUT FORMAT:
You MUST respond in valid JSON format with the following keys:
{{
  "reply": "Your conversational response here. Be professional and helpful.",
  "recommendations": [
    {{
      "name": "Exact assessment name",
      "url": "Verbatim URL from context",
      "test_type": "A for Ability/Aptitude, P for Personality/Behavior, K for Knowledge/Skills, S for Simulations"
    }}
  ],
  "end_of_conversation": true/false
}}

Rules:
- 'recommendations' should be EMPTY [] while you are still clarifying or if you are refusing.
- 'end_of_conversation' is true ONLY when a final shortlist has been accepted by the user or the query is refused.
- NEVER recommend anything not present in the CONTEXT.
- All URLs must be verbatim from the context.
"""

def search_catalog(query: str, n: int = 12) -> list[dict]:
    """Queries the ChromaDB collection and returns top-n results."""
    results = collection.query(
        query_texts=[query],
        n_results=n
    )
    
    formatted_results = []
    if results['ids'] and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "name": results['metadatas'][0][i]['name'],
                "url": results['metadatas'][0][i]['url'],
                "test_type": results['metadatas'][0][i]['test_type'],
                "chunk": results['documents'][0][i]
            })
    return formatted_results

def call_llm(messages: list[dict], catalog_context: str) -> dict:
    """Calls the Groq LLM with the provided messages and context."""
    client_llm = get_groq_client()
    if not client_llm:
        print("Error: GROQ_API_KEY not set.")
        return {
            "reply": "I'm currently unable to access my advisor logic. Please ensure the GROQ_API_KEY is configured.",
            "recommendations": [],
            "end_of_conversation": False
        }

    full_system_prompt = SYSTEM_PROMPT.format(catalog_context=catalog_context)
    
    # Prepend system prompt to the messages
    llm_messages = [{"role": "system", "content": full_system_prompt}] + messages
    
    try:
        completion = client_llm.chat.completions.create(
            messages=llm_messages,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1000,
        )
        
        response_content = completion.choices[0].message.content
        
        # Robustly extract JSON block using regex
        json_match = re.search(r'(\{.*\})', response_content, re.DOTALL)
        if json_match:
            clean_content = json_match.group(1)
        else:
            clean_content = response_content # Fallback to original

        try:
            return json.loads(clean_content)
        except json.JSONDecodeError:
            print(f"Error: Could not parse JSON. Content: {clean_content[:100]}...")
            return {
                "reply": "I apologize, but I encountered an error processing my response. How else can I help you with SHL assessments?",
                "recommendations": [],
                "end_of_conversation": False
            }
            
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {
            "reply": "I'm sorry, I'm having trouble connecting to my knowledge base right now. Please try again in a moment.",
            "recommendations": [],
            "end_of_conversation": False
        }

def run_agent(messages: list[dict]) -> dict:
    """Main entry point for the agent logic."""
    # Build retrieval query from last 3 user messages
    user_messages = [m['content'] for m in messages if m['role'] == 'user']
    retrieval_query = " ".join(user_messages[-3:])
    
    # Get top 12 hits from the catalog
    hits = search_catalog(retrieval_query, n=12)
    
    # Create a string representation of the context for the LLM
    catalog_context = "\n---\n".join([h['chunk'] for h in hits])
    
    # Call the LLM
    agent_response = call_llm(messages, catalog_context)
    
    # POST-FILTERING
    retrieved_urls = {h['url'] for h in hits}
    filtered_recs = []
    
    for rec in agent_response.get('recommendations', []):
        if rec.get('url') in retrieved_urls:
            filtered_recs.append(rec)
            
    # Cap at 10
    agent_response['recommendations'] = filtered_recs[:10]
    
    return agent_response

if __name__ == "__main__":
    # Example test run (requires GROQ_API_KEY)
    test_messages = [{"role": "user", "content": "I need a personality test for a senior marketing manager role."}]
    if os.getenv("GROQ_API_KEY"):
        result = run_agent(test_messages)
        print(json.dumps(result, indent=2))
    else:
        print("GROQ_API_KEY not found. Skipping test run.")

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
1. CLARIFY: If the user's request is truly vague with NO role, skill, or job title (e.g., just "I need an assessment"), ask 1-2 targeted questions. IMPORTANT: If the user message contains a job title (e.g., "software engineer", "manager") OR a specific skill (e.g., "Java", "Python", "leadership", "sales", "AWS"), that is SUFFICIENT context to recommend immediately — do NOT ask for clarification first.
   - FAST-TRACK RULE: If the user's message is longer than 30 words OR contains the phrase 'job description', treat it as sufficient context and go directly to recommendations on that turn. Do not ask clarifying questions.
2. RECOMMEND: Once you have a role or skill, recommend up to 10 assessments. For each, provide the EXACT name and URL from the context. Always prefer recommending MORE relevant assessments (up to 10) over fewer. Include both role-specific AND general cognitive/personality tests when the role involves managing people or complex decisions.
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
- 'recommendations' should be EMPTY [] ONLY while clarifying a truly vague query, or if refusing.
- 'end_of_conversation' is true ONLY when a final shortlist has been accepted by the user or the query is refused.
- NEVER recommend anything not present in the CONTEXT.
- All URLs must be verbatim from the context.
"""

def search_catalog(query: str, n: int = 12, where: dict = None) -> list[dict]:
    """Queries the ChromaDB collection and returns top-n results with optional filtering."""
    results = collection.query(
        query_texts=[query],
        n_results=n,
        where=where
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
            timeout=25
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

def get_type_boosted_results(query: str, n: int = 15) -> list[dict]:
    """Fetches semantic results and boosts specific test types based on keywords."""
    # Main semantic results
    base_hits = search_catalog(query, n=n)
    
    query_lower = query.lower()
    boost_hits = []
    
    # Leadership/Management Boost
    mgmt_keywords = ["manage", "lead", "team", "stakeholder", "director", "head of", "executive", "supervisor"]
    if any(kw in query_lower for kw in mgmt_keywords):
        # Boost Personality (P) and Ability (A)
        boost_hits += search_catalog(query, n=5, where={"test_type": "P"})
        boost_hits += search_catalog(query, n=5, where={"test_type": "A"})
        
    # Technical Boost
    tech_keywords = ["technical", "developer", "engineer", "code", "programming", "software", "api", "data", "cloud"]
    if any(kw in query_lower for kw in tech_keywords):
        # Boost Knowledge (K)
        boost_hits += search_catalog(query, n=5, where={"test_type": "K"})
        
    # Merge and deduplicate
    seen_urls = set()
    merged = []
    for h in boost_hits + base_hits:
        if h['url'] not in seen_urls:
            merged.append(h)
            seen_urls.add(h['url'])
            
    return merged[:25] # Cap at 25 unique items for LLM context

def _build_structured_query(user_messages: list[str]) -> str:
    """Extracts structured signals from user messages for richer retrieval."""
    all_text = " ".join(user_messages).lower()

    # Signal patterns
    role_keywords = [
        "engineer", "developer", "manager", "analyst", "scientist", "designer",
        "director", "executive", "sales", "support", "nurse", "accountant",
        "cashier", "receptionist", "data", "cloud", "devops", "frontend", "backend"
    ]
    seniority_keywords = [
        "senior", "junior", "entry", "mid", "lead", "principal", "director",
        "executive", "graduate", "intern", "head of"
    ]
    skill_keywords = [
        "java", "python", "javascript", "c++", "c#", "sql", "aws", "azure",
        "docker", "kubernetes", "react", "angular", "node", "spark", "hadoop",
        "machine learning", "leadership", "personality", "communication",
        "agile", "scrum", "git", "devops", "cloud", "data science", "statistics"
    ]

    role = next((kw for kw in role_keywords if kw in all_text), None)
    seniority = next((kw for kw in seniority_keywords if kw in all_text), None)
    skills = [kw for kw in skill_keywords if kw in all_text]

    # If we have at least a role or a skill, build a structured query
    if role or skills:
        parts = []
        if role: parts.append(f"role: {role}")
        if seniority: parts.append(f"level: {seniority}")
        if skills: parts.append(f"skills: {' '.join(skills[:4])}")
        return " ".join(parts)

    # Fallback: concatenate all user messages
    return " ".join(user_messages)


def run_agent(messages: list[dict]) -> dict:
    """Main entry point for the agent logic."""
    user_messages = [m['content'] for m in messages if m['role'] == 'user']

    # --- PASS 1: Structured query with Type Boosting ---
    structured_query = _build_structured_query(user_messages)
    structured_hits = get_type_boosted_results(structured_query, n=15)

    # --- PASS 2: Latest user intent with Type Boosting ---
    latest_hits = get_type_boosted_results(user_messages[-1], n=10)

    # Merge all passes, deduplicate by URL
    # Priority: latest intent > structured
    seen_urls = set()
    hits = []
    for h in latest_hits + structured_hits:
        if h['url'] not in seen_urls:
            hits.append(h)
            seen_urls.add(h['url'])

    # Keep top 25 unique hits for the LLM
    hits = hits[:25]

    # Build context string for single LLM call
    catalog_context = "\n---\n".join([h['chunk'] for h in hits])

    # --- Fast-track Check ---
    last_user_msg = user_messages[-1]
    fast_track = False
    if len(last_user_msg.split()) > 30 or "job description" in last_user_msg.lower():
        fast_track = True
        # Inject a hint as a temporary system-like message
        messages = messages + [{"role": "system", "content": "HINT: The user has provided a detailed message or a job description. Recommend assessments immediately."}]

    # --- Single LLM call with full enriched context ---
    agent_response = call_llm(messages, catalog_context)

    # POST-FILTERING: only keep items actually in retrieved context
    retrieved_urls = {h['url'] for h in hits}
    filtered_recs = [
        rec for rec in agent_response.get('recommendations', [])
        if rec.get('url') in retrieved_urls
    ]

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

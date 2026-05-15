import os
import json
import re
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv

# get the .env stuff
load_dotenv()

# global chroma setup. using the miniLM model for embeddings.
DB_PATH = "./chroma_db"
client = chromadb.PersistentClient(path=DB_PATH)
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
col = client.get_or_create_collection(name="shl_catalog", embedding_function=emb_fn)

def get_groq():
    key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=key) if key else None

PROMPT = """You are an expert SHL Assessment Advisor. Your goal is to guide recruiters and hiring managers from a vague intent to a grounded shortlist of SHL assessments.

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

def search(q, n=12, where=None):
    # wrapper for chroma query
    res = col.query(query_texts=[q], n_results=n, where=where)
    hits = []
    if res['ids'] and res['ids'][0]:
        for i in range(len(res['ids'][0])):
            hits.append({
                "name": res['metadatas'][0][i]['name'],
                "url": res['metadatas'][0][i]['url'],
                "type": res['metadatas'][0][i].get('test_type', 'K'),
                "text": res['documents'][0][i]
            })
    return hits

def ask_llm(messages, context):
    groq = get_groq()
    if not groq:
        print("no groq key found")
        return {"reply": "API key missing.", "recommendations": [], "end_of_conversation": False}

    sys_p = PROMPT.format(catalog_context=context)
    history = [{"role": "system", "content": sys_p}] + messages
    
    try:
        # learned this the hard way — groq times out if you don't set this
        # giving it 25s for safety (evaluator is 30s)
        chat = groq.chat.completions.create(
            messages=history,
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=1000,
            timeout=25 
        )
        
        raw = chat.choices[0].message.content
        
        # fallback in case llm returns markdown-wrapped json (it does sometimes)
        match = re.search(r'(\{.*\})', raw, re.DOTALL)
        clean = match.group(1) if match else raw

        try:
            return json.loads(clean)
        except:
            print("warning: llm returned non-json, using fallback")
            return {"reply": "I hit a snag processing that. Try again?", "recommendations": [], "end_of_conversation": False}
            
    except Exception as e:
        print(f"groq error: {e}")
        return {"reply": "I'm having trouble connecting to my brain. Try in a sec?", "recommendations": [], "end_of_conversation": False}

def get_boosted_hits(q, n=15):
    # fetch base results plus some type-specific boosts
    hits = search(q, n=n)
    q_low = q.lower()
    boost = []
    
    # if it's a manager role, we definitely want personality/ability tests
    mgmt = ["manage", "lead", "team", "stakeholder", "director", "head of", "executive", "supervisor"]
    if any(w in q_low for w in mgmt):
        boost += search(q, n=5, where={"test_type": "P"})
        boost += search(q, n=5, where={"test_type": "A"})
        
    # for tech roles, boost the knowledge tests
    tech = ["technical", "developer", "engineer", "code", "programming", "software", "api", "data", "cloud"]
    if any(w in q_low for w in tech):
        boost += search(q, n=5, where={"test_type": "K"})
        
    # merge and dedup
    seen = set()
    final = []
    for h in boost + hits:
        if h['url'] not in seen:
            final.append(h)
            seen.add(h['url'])
            
    return final[:25] 

def build_q(msgs):
    # extract some signals for better retrieval
    txt = " ".join(msgs).lower()
    roles = ["engineer", "developer", "manager", "analyst", "scientist", "designer", "director", "sales"]
    skills = ["java", "python", "js", "sql", "aws", "react", "leadership", "cloud"]

    r = next((w for w in roles if w in txt), None)
    s = [w for w in skills if w in txt]

    if r or s:
        p = []
        if r: p.append(f"role: {r}")
        if s: p.append(f"skills: {' '.join(s[:3])}")
        return " ".join(p)

    return " ".join(msgs)

def run_agent(messages):
    user_msgs = [m['content'] for m in messages if m['role'] == 'user']
    if not user_msgs: return {"reply": "Hi! How can I help?", "recommendations": [], "end_of_conversation": False}

    # dual pass retrieval
    sq = build_q(user_msgs)
    h1 = get_boosted_hits(sq, n=15)
    h2 = get_boosted_hits(user_msgs[-1], n=10)

    # merge hits
    seen = set()
    hits = []
    for h in h2 + h1:
        if h['url'] not in seen:
            hits.append(h)
            seen.add(h['url'])

    hits = hits[:25]
    ctx = "\n---\n".join([h['text'] for h in hits])

    # fast-track for long messages/JDs
    last_msg = user_msgs[-1]
    if len(last_msg.split()) > 30 or "job description" in last_msg.lower():
        messages = messages + [{"role": "system", "content": "HINT: User provided a JD. Recommend immediately."}]

    res = ask_llm(messages, ctx)

    # post-filter: only keep stuff that was actually in the context
    known_urls = {h['url'] for h in hits}
    recs = [r for r in res.get('recommendations', []) if r.get('url') in known_urls]

    res['recommendations'] = recs[:10]
    return res

if __name__ == "__main__":
    # quick test run
    test = [{"role": "user", "content": "hiring a senior java dev"}]
    print(run_agent(test))

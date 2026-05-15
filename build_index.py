import json
import os
import chromadb
from chromadb.utils import embedding_functions

def run_indexing():
    # check if we actually have data to index
    if not os.path.exists('catalog.json'):
        print("no catalog.json found. you gotta run scraper.py first.")
        return

    with open('catalog.json', 'r') as f:
        data = json.load(f)

    # set up chroma locally
    db = chromadb.PersistentClient(path="./chroma_db")
    
    # using miniLM because it's fast and small. 
    # if this gets slow we might need something beefier.
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    col = db.get_or_create_collection(
        name="shl_catalog",
        embedding_function=emb_fn
    )

    ids, docs, metas = [], [], []

    print(f"indexing {len(data)} items...")

    for i, item in enumerate(data):
        name = item.get('name', 'N/A')
        # some versions of the scraper used 'url', some used 'link'
        url = item.get('link') or item.get('url') or 'N/A'
        
        # figuring out the type from the keys list
        # default to K (knowledge) if we can't find a better match
        keys = item.get('keys', [])
        t_type = 'K' 
        if 'Personality & Behavior' in keys:
            t_type = 'P'
        elif 'Ability & Aptitude' in keys:
            t_type = 'A'
        elif 'Simulations' in keys:
            t_type = 'S'
            
        desc = item.get('description', '')
        levels = item.get('job_levels', 'N/A')
        dur = item.get('duration', 'N/A')

        # building a big string for the vector search to chew on
        chunk = (
            f"Assessment: {name}\n"
            f"Type: {t_type}\n"
            f"Description: {desc}\n"
            f"Levels: {levels}\n"
            f"Time: {dur}\n"
            f"URL: {url}"
        )

        ids.append(f"id_{i}")
        docs.append(chunk)
        metas.append({
            "name": str(name),
            "url": str(url),
            "test_type": str(t_type)
        })

    # write it all to the db
    col.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"done. indexed {col.count()} items.")

# just for testing
def test_search(q, n=3):
    db = chromadb.PersistentClient(path="./chroma_db")
    fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    col = db.get_collection(name="shl_catalog", embedding_function=fn)
    
    res = col.query(query_texts=[q], n_results=n)
    
    hits = []
    for i in range(len(res['ids'][0])):
        hits.append({
            "name": res['metadatas'][0][i]['name'],
            "url": res['metadatas'][0][i]['url'],
            "type": res['metadatas'][0][i]['test_type']
        })
    return hits

if __name__ == "__main__":
    run_indexing()
    
    print("\nquick check:")
    for h in test_search("java developer", n=2):
        print(f"- {h['name']} [{h['type']}]")

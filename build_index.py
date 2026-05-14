import json
import os
import chromadb
from chromadb.utils import embedding_functions

def build_index():
    # 1. Reads catalog.json
    if not os.path.exists('catalog.json'):
        print("Error: catalog.json not found. Run scraper.py first.")
        return

    with open('catalog.json', 'r') as f:
        catalog = json.load(f)

    # 2. Creates a ChromaDB persistent collection
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 3. Uses SentenceTransformer "all-MiniLM-L6-v2" as the embedding function
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    collection = client.get_or_create_collection(
        name="shl_catalog",
        embedding_function=embedding_func
    )

    ids = []
    documents = []
    metadatas = []

    print(f"Indexing {len(catalog)} products...")

    # 4. For each product, builds a rich text chunk
    for i, item in enumerate(catalog):
        name = item.get('name', 'N/A')
        url = item.get('url', 'N/A')
        test_type = item.get('test_type', 'K')
        description = item.get('description', '')
        job_levels = item.get('job_levels', 'N/A')
        duration = item.get('duration', 'N/A')

        chunk = (
            f"Assessment: {name}\n"
            f"Type: {test_type}\n"
            f"Description: {description}\n"
            f"Job levels: {job_levels}\n"
            f"Duration: {duration}\n"
            f"URL: {url}"
        )

        ids.append(f"prod_{i}")
        documents.append(chunk)
        # 5. Upserts all chunks with metadata (strings)
        metadatas.append({
            "name": str(name),
            "url": str(url),
            "test_type": str(test_type),
            "duration": str(duration)
        })

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    # 6. Prints how many items were indexed
    print(f"Successfully indexed {collection.count()} items into 'shl_catalog'.")

def search_catalog(query, n=10):
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name="shl_catalog", embedding_function=embedding_func)
    
    results = collection.query(
        query_texts=[query],
        n_results=n
    )
    
    formatted_results = []
    for i in range(len(results['ids'][0])):
        formatted_results.append({
            "name": results['metadatas'][0][i]['name'],
            "url": results['metadatas'][0][i]['url'],
            "test_type": results['metadatas'][0][i]['test_type'],
            "chunk": results['documents'][0][i]
        })
    
    return formatted_results

if __name__ == "__main__":
    build_index()
    
    # Quick verification search
    print("\nVerifying with a sample query: 'leadership personality assessment'")
    sample_results = search_catalog("leadership personality assessment", n=3)
    for res in sample_results:
        print(f"- {res['name']} ({res['test_type']})")

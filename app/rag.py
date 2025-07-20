import faiss
import pickle
import os

from sentence_transformers import SentenceTransformer
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# === Load on startup ===
DATA_DIR = "data"
INDEX_PATH = os.path.join(DATA_DIR, "faiss_products.index")
META_PATH = os.path.join(DATA_DIR, "faiss_products_metadata.pkl")

print("🔄 Loading vector store and metadata...")
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_search(query: str, top_k: int = 3) -> List[dict]:
    embedding = model.encode([query])
    D, I = index.search(embedding, top_k)
    print("Indexes returned:", I)
    print("Raw metadata:", [metadata[i] for i in I[0]])  # Debug
    results = [clean_result(metadata[i]) for i in I[0]]
    return results

def clean_result(r: dict) -> dict:
    if isinstance(r, list) and len(r) > 0:
        r = r[0]  # Unwrap if list of one dict

    return {
        "name": r.get("name", "Unknown Product"),
        "price": r.get("price", "N/A"),
        "variations": r.get("variations", []),
        "product_info": r.get("product_info") if r.get("product_info") else ["No product info available"],
        "measurements": r.get("measurements") if r.get("measurements") else {"Height": "N/A", "Volume": "N/A"},
        "materials": r.get("materials") if r.get("materials") else {"Note": "No material info available"},
        "url": r.get("url", "#")
    }

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY")
)

def summarize_results(query: str, results: List[dict]) -> str:
    print("Results type:", type(results))
    print("First result type:", type(results[0]))
    print("First result keys:", results[0].keys())

    content = "\n".join([
        f"- {r.get('name', 'Unknown')} | "
        f"{', '.join(r.get('product_info', ['No product info available']))} | "
        f"Variations: {', '.join(r.get('variations', ['N/A']))} | "
        f"Price: {r.get('price', 'N/A')} | "
        f"Volume: {r.get('measurements', {}).get('Volume', 'N/A')} | "
        f"Height: {r.get('measurements', {}).get('Height', 'N/A')} | "
        f"Material: {', '.join(r.get('materials', {}).keys()) if r.get('materials') else 'No material info available'}"
        for r in results
    ])

    prompt = f"""
You are a helpful assistant for ZUS Coffee product discovery. Answer the question below based on this product info:

{content}

User Question: {query}

Answer:
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


import os
import json
import glob
import faiss
import pickle

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === SETTINGS ===
DATA_DIR = "data"
VECTOR_DB_PATH = os.path.join(DATA_DIR, "faiss_products.index")
METADATA_PATH = os.path.join(DATA_DIR, "faiss_products_metadata.pkl")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Small + fast

# === LOAD LATEST PRODUCT FILE ===
def get_latest_json_file():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "products_*.json")), reverse=True)
    return files[0] if files else None

def create_text_chunks(product):
    """Create a combined string for semantic search."""
    parts = [product.get("name", "")]
    parts += product.get("variations", [])
    parts += product.get("product_info", [])
    
    # Add key-value pairs like "Volume: 500ml", "Material: SUS304"
    for section in ["measurements", "materials"]:
        section_dict = product.get(section, {})
        for k, v in section_dict.items():
            parts.append(f"{k}: {v}")

    return ". ".join(filter(None, parts))

def main():
    latest_file = get_latest_json_file()
    if not latest_file:
        print("❌ No product JSON file found in 'data/' folder.")
        return

    print(f"📄 Loading: {latest_file}")
    with open(latest_file, "r", encoding="utf-8") as f:
        products = json.load(f)

    print(f"🧠 Generating embeddings using {EMBED_MODEL_NAME}...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = []
    metadata = []
    for p in products:
        chunk = create_text_chunks(p)
        texts.append(chunk)
        metadata.append(p)

    embeddings = model.encode(texts, show_progress_bar=True)

    print("💾 Saving FAISS index and metadata...")
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, VECTOR_DB_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print("✅ Vector store created and saved!")

if __name__ == "__main__":
    main()

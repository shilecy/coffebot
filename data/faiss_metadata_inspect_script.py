import pickle

with open("data/faiss_products_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Print the number of records
print(f"✅ Total records: {len(metadata)}\n")

# Print first 3 records to inspect format
for i, record in enumerate(metadata[:5]):
    print(f"--- Record {i+1} ---")
    for k, v in record.items():
        print(f"{k}: {v}")
    print()

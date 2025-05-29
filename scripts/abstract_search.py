import json
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "/home/ec2-user/arxiv-copilot/embeddings/1000000/faiss_index.index"
META_PATH = "/home/ec2-user/arxiv-copilot/embeddings/1000000/meta.json"
# INDEX_PATH = "/Users/luojidong/程式/arxiv-copilot/embeddings/1000000/faiss_index.index"
# META_PATH = "/Users/luojidong/程式/arxiv-copilot/embeddings/1000000/meta.json"

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "r") as f:
    metadata = json.load(f)

# Function to search for abstracts based on a query
# Returns the top_k most similar abstracts
def search_abstracts(query, top_k=3):
    query_vec = model.encode([query])
    _, I = index.search(query_vec, top_k)
    return [metadata[i] for i in I[0]]


if __name__ == "__main__":
    results = search_abstracts("biology dolphin", top_k=10)
    for r in results:
        print(f"Title: {r['title']}")
        print(f"ID: {r['id']}")
        print(f"Authors: {r['authors']}")
        print(f"Abstract: {r['abstract']}")
        print("-" * 80)
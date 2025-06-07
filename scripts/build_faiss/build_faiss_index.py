import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
import os

# === Settings ===
JSON_PATH = "/Users/luojidong/程式/arxiv-copilot/data/filter/filtered_arxiv_cs_v2.jsonl"  # update path if needed
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SAVE_DIR = "/Users/luojidong/程式/arxiv-copilot/embeddings/filterv2"
MAX_DOCS = None  # change to None to load entire d


# === Step 1: Load JSON abstracts ===
abstracts = []
paper_meta = []

with open(JSON_PATH, 'r') as f:
    for line in tqdm(f, desc="Loading abstracts"):
        record = json.loads(line)
        if record.get("abstract"):
            clean_abstract = record["abstract"].strip().replace('\n', ' ')
            abstracts.append(clean_abstract)
            paper_meta.append({
                "id": record["id"],
                "title": record["title"],
                "authors": record["authors"],
                "categories": record["categories"],
                "abstract": clean_abstract,
            })
            if MAX_DOCS and len(abstracts) >= MAX_DOCS:
                break

print(f"✅ Loaded {len(abstracts)} abstracts.")

# === Step 2: Generate embeddings ===
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(abstracts, show_progress_bar=True, batch_size=64)
print(f"embeddings shape: {embeddings.shape}") # [100000, 384]

# === Step 3: Save artifacts ===
os.makedirs(SAVE_DIR, exist_ok=True)
np.save(f"{SAVE_DIR}/abstracts.npy", embeddings)
with open(f"{SAVE_DIR}/meta.json", "w") as f:
    json.dump(paper_meta, f)

# === Step 4: Build FAISS index ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings) # Add vectors
faiss.write_index(index, f"{SAVE_DIR}/faiss_index.index")

print(f"✅ FAISS index built with {len(embeddings)} entries.") # 100000
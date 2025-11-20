# ingest_documents.py
import os
import json
from datetime import datetime
import boto3
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
AWS_REGION = os.getenv("AWS_REGION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
SNAPSHOT_PATH = os.getenv("SNAPSHOT_PATH")  # path to arxiv-metadata-oai-snapshot.json
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
MAX_DOCS = os.getenv("MAX_DOCS") 
MAX_DOCS = int(MAX_DOCS) if MAX_DOCS is not None else None
VECTOR_BUCKET_NAME = os.getenv("VECTOR_BUCKET_NAME")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME")
VECTOR_ARN_NAME = os.getenv("VECTOR_ARN_NAME")
INDEX_ARN_NAME = os.getenv("INDEX_ARN_NAME")
START_LINE = 55800


openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
s3vec = boto3.client("s3vectors", region_name=AWS_REGION)

# ---------- parsing & text building ----------

def parse_raw_record(line: str) -> dict | None:
    """Parse one JSON line from the arxiv snapshot into a document dict."""
    try:
        raw = json.loads(line)
    except json.JSONDecodeError:
        return None

    paper_id = raw.get("id")
    if not paper_id:
        return None

    title = (raw.get("title") or "").replace("\n", " ").strip()
    abstract = (raw.get("abstract") or "").strip()

    # authors: 優先用 authors_parsed
    if raw.get("authors_parsed"):
        authors = [" ".join([p for p in parts if p]) for parts in raw["authors_parsed"]]
    else:
        authors = [a.strip() for a in (raw.get("authors") or "").split(",") if a.strip()]

    categories = [c.strip() for c in (raw.get("categories") or "").split() if c.strip()]

    # year: from first version.created or update_date
    year = None
    versions = raw.get("versions") or []
    if versions:
        created_str = versions[0].get("created")
        if created_str:
            try:
                year = datetime.strptime(created_str, "%a, %d %b %Y %H:%M:%S %Z").year
            except Exception:
                pass
    if year is None and raw.get("update_date"):
        try:
            year = int(raw["update_date"][:4])
        except Exception:
            year = None

    return {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "categories": categories,
        "year": year,
    }


def build_embedding_text(doc: dict) -> str:
    authors_str = ", ".join(doc.get("authors") or [])
    categories_str = ", ".join(doc.get("categories") or [])

    parts = [doc.get("title", "")]
    if authors_str:
        parts.append(f"Authors: {authors_str}")
    if categories_str:
        parts.append(f"Categories: {categories_str}")
    if doc.get("abstract"):
        parts.append("Abstract:\n" + doc["abstract"])

    return "\n\n".join(parts).strip()


# ---------- Titan embedding ----------

def embed_with_openai(text: str) -> list[float]:
    resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return resp.data[0].embedding

def put_vectors_batch(vectors: list[dict]):
    """
    vectors: each element is like this:
    {
        "key": "paper:0704.0001",
        "data": {'float32': [...,]},  
        "metadata": {
            "paper_id": "...",
            "title": "...",
            ...
        }
    }
    """
    if not vectors:
        return
    resp = s3vec.put_vectors(
        indexArn=INDEX_ARN_NAME,
        vectors=vectors,
    )

def ingest_snapshot():
    count = 0
    batch: list[dict] = []

    with open(SNAPSHOT_PATH, "r") as f:
        for i, line in enumerate(f, start=1):
            if i <= START_LINE:
                continue
            line = line.strip()
            if not line:
                continue

            doc = parse_raw_record(line)
            if not doc or not doc.get("abstract") or not doc.get("title"):
                continue  # skip garbage

            text = build_embedding_text(doc)
            embedding = embed_with_openai(text)
            if not embedding:
                continue

            vector = {
                "key": f"paper:{doc['paper_id']}",
                "data": {
                    "float32": embedding
                },
                "metadata": {
                    "paper_id": doc["paper_id"],
                    "title": doc["title"],
                    "authors": doc.get("authors", []),
                    "categories": doc.get("categories", []),
                    "year": doc.get("year"),
                    "abstract": doc.get("abstract")
                }
            }
            
            batch.append(vector)
            count += 1

            if len(batch) >= BATCH_SIZE:
                print(f"Uploading batch, total so far: {count}")
                put_vectors_batch(batch)
                batch = []

            if MAX_DOCS is not None and count >= MAX_DOCS:
                break

    if batch:
        print(f"Uploading final batch, total: {count}")
        put_vectors_batch(batch)


if __name__ == "__main__":
    print(f"Start ingest to S3 Vectors index={VECTOR_INDEX_NAME}")
    ingest_snapshot()
    print("Done.")

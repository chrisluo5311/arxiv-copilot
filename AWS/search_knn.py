import os
import boto3
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
VECTOR_BUCKET_NAME = os.getenv("VECTOR_BUCKET_NAME")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME")
VECTOR_ARN_NAME = os.getenv("VECTOR_ARN_NAME")
INDEX_ARN_NAME = os.getenv("INDEX_ARN_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
s3vec = boto3.client("s3vectors", region_name=AWS_REGION)


def embed_with_openai(text: str) -> list[float]:
    resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return resp.data[0].embedding

def search_papers(query: str, top_k: int = 5):
    query_vec = embed_with_openai(query)

    resp = s3vec.query_vectors(
        indexArn=INDEX_ARN_NAME,
        topK=top_k,
        queryVector={
            "float32": query_vec,
        },
        returnMetadata=True,
        returnDistance=True,
    )
    # print(f"resp: {resp}")

    results = []
    distance_metric = resp.get("distanceMetric")  # 'cosine' or 'euclidean'
    # resp structure is like: {"vectors": [{"key": ..., "data": {"float32": [...,]}, "metadata": {...}, "distance": ...}], 'distanceMetric': 'euclidean'|'cosine'}
    for v in resp.get("vectors", []):
        meta = v.get("metadata", {})
        distance = v.get("distance")
        if distance is not None:
            if distance_metric == "cosine":
                # cosine distance = 1 - cosine_sim → score bigger is better
                score = 1.0 - distance
            else:
                # euclidean etc. -> distance smaller is better, here is a simple conversion
                score = 1.0 / (1.0 + distance)
        results.append({
            "paper_id": meta.get("paper_id") or v.get("key"),
            "title": meta.get("title"),
            "abstract": meta.get("abstract"),
            "authors": meta.get("authors", []),
            "categories": meta.get("categories", []),
            "year": meta.get("year"),
            "score": score,
            "distance": distance,
        })

    return results


if __name__ == "__main__":
    q = "AI agent"
    results = search_papers(q, top_k=10)
    for i, r in enumerate(results, 1):
        print(f"\n=== Result {i} — score {r['score']} ===")
        print(f"{r['paper_id']}: {r['title']}")
        print("Categories:", ", ".join(r.get("categories", [])))
        print("Authors:", ", ".join(r.get("authors", [])))
        print("Abstract:", r["abstract"][:400], "...")

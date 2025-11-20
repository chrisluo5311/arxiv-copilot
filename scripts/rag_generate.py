import os
from openai import OpenAI
from dotenv import load_dotenv
from scripts.storage_s3 import (
    read_text_from_s3,
    S3_CHUNK_PREFIX,
    S3_UPLOAD_CHUNK_PREFIX,
)
import math
# ------------------ Configuration ------------------ #
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------ Titan embedding ------------------ #
def embed_with_openai(text: str) -> list[float]:
    resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return resp.data[0].embedding

def cosine_sim(a: list[float], b: list[float]) -> float:
    # manually calculate cosine similarity, avoid pulling numpy / torch
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def _chunk_text(full_text: str, chunk_size: int = 300, stride: int = 100) -> list[str]:
    words = full_text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    chunks = [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words) - chunk_size + 1, stride)
    ]
    return chunks

def load_and_chunk(arxiv_id, chunk_size=300, stride=100):
    """
    Load and chunk the PDF from S3
    Args:
        arxiv_id: the arXiv ID of the paper
        chunk_size: the size of the chunks
        stride: the stride of the chunks
    Returns:
        chunks: the chunks of the PDF
    """
    key = S3_CHUNK_PREFIX + f"{arxiv_id}.txt"
    full_text = read_text_from_s3(key)
    return _chunk_text(full_text, chunk_size=chunk_size, stride=stride)

def load_uploaded_file_and_chunk(file_name, chunk_size=300, stride=100):
    """
    Load and chunk the uploaded file
    Args:
        file_name: the name of the uploaded file
        chunk_size: the size of the chunks
        stride: the stride of the chunks
    Returns:
        chunks: the chunks of the uploaded file
    """
    key = S3_UPLOAD_CHUNK_PREFIX + f"{file_name}.txt"
    full_text = read_text_from_s3(key)
    return _chunk_text(full_text, chunk_size=chunk_size, stride=stride)

def retrieve_top_chunks(query, chunks, top_k=10):
    """
    Retrieve the top chunks based on the query
    Args:
        query: the query
        chunks: the chunks
        top_k: the number of top chunks
    Returns:
        top_chunks: the top chunks
    """
    if not chunks:
        raise ValueError("No chunks available for retrieval.")

    query_emb = embed_with_openai(query)

    scored = []
    for ch in chunks:
        ch_emb = embed_with_openai(ch)
        score = cosine_sim(query_emb, ch_emb)
        scored.append((score, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [ch for score, ch in scored[:top_k]]
    return top

def standalone_answer(query, chunks, model_name, base64_image, api_key=None):
    """
    Generate the answer based on the query and the chunks   
    Args:
        query: the query
        chunks: the chunks
        model_name: the name of the model
        base64_image: the base64 image
        api_key: the API key
    Returns:
        answer: the answer
    """
    client = OpenAI(api_key=api_key)
    context = "\n\n".join(chunks)
    prompt = f"""You are a helpful AI assistant specialized in scientific research.
            You are given a user question and some context extracted from a scientific paper.
            
            Context:
            \"\"\"
            {context}
            \"\"\"
            
            Question: {query}
            Answer:"""

    # Build content conditionally based on whether image is provided
    if base64_image:
        # Multimodal content: array with text and image
        user_content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        ]
    else:
        # Text-only content: simple string
        user_content = prompt
    
    response = client.chat.completions.create(
        model=model_name,
        temperature=1,
        messages=[
            {
                "role": "developer",
                "content": "You are a research assistant AI."
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    )
    return response.choices[0].message.content.strip()

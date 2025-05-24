import os
from openai import OpenAI
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Configuration ------------------ #
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# o4-mini = 200,000 context window # 100,000 max output tokens
# gpt-3.5-turbo = 16,385 context window # 4,096 max output tokens
# gpt-4o = 128,000 context window # 16,384 max output tokens
# chatgpt-4o (currently used in ChatGPT) = 128,000 context window # 16,384 max output tokens
MODEL_NAME = "gpt-3.5-turbo"
MAX_CONTEXT_TOKENS = 12_000           # leave room for the question & response
model = SentenceTransformer("all-MiniLM-L6-v2")
# --------------------------------------------------- #

# Try chunk_size = 300–400, stride = 100–200
#  - smaller chunk_size for short question-answering
#  - larger chunk_size for summarization or multi-section questions
def load_and_chunk(arxiv_id, chunk_dir="../pdf_chunks", chunk_size=300, stride=100):
    path = os.path.join(chunk_dir, f"{arxiv_id}.txt") # ../pdf_chunks/0704.0001.txt
    if not os.path.exists(path):
        raise FileNotFoundError(f"arXiv file not found: {path}")
    with open(path, "r") as f:
        full_text = f.read()

    # Split into chunks of N words
    words = full_text.split()
    # Overlapping: [0,299], [100,399], [200,499], ...
    # Better continuity between chunks
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words) - chunk_size + 1, stride)]
    return chunks

def retrieve_top_chunks(query, chunks, top_k=5):
    chunk_embeddings = model.encode(chunks)
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def generate_answer(query, chunks, model_name, api_key=None):
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

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.3,
        messages=[
            {
                "role": "developer",
                "content": "You are a research assistant AI."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content.strip()

# Example
if __name__ == "__main__":
    arxiv_id = "0704.0001"
    query = "How does the paper calculate diphoton production cross-sections?"
    chunks = load_and_chunk(arxiv_id)
    top_chunks = retrieve_top_chunks(query, chunks, top_k=5)
    answer = generate_answer(query, top_chunks, MODEL_NAME, api_key=os.getenv("OPENAI_API_KEY"))

    print("🔎 Answer:")
    print(answer)

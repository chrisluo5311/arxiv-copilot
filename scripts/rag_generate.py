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
MODEL_NAME = "gpt-4o"
MAX_CONTEXT_TOKENS = 12_000  # leave room for the question & response
model = SentenceTransformer("all-MiniLM-L6-v2")

# /Users/luojidong/程式/arxiv-copilot/scripts
current_dir = os.path.dirname(os.path.abspath(__file__)) 
# /Users/luojidong/程式/arxiv-copilot/scripts/pdf_chunks
chunk_dir = os.path.join(current_dir, "pdf_chunks") 
# /Users/luojidong/程式/arxiv-copilot/scripts/pdf_chunks
upload_file_chunk_dir = os.path.join(current_dir, "upload_file_chunks") 
# --------------------------------------------------- #

def load_and_chunk(arxiv_id, chunk_size=300, stride=100):
    """
    Overlapping: [0,299], [100,399], [200,499] for better continuity between chunks
    - smaller chunk_size for short question-answering
    - larger chunk_size for summarization or multi-section questions
    - stride: overlap between chunks
    """
    # /Users/luojidong/程式/arxiv-copilot/scripts/pdf_chunks/0704.0001.txt
    path = os.path.join(chunk_dir, f"{arxiv_id}.txt") 
    if not os.path.exists(path):
        raise FileNotFoundError(f"arXiv file not found: {path}")
    with open(path, "r") as f:
        full_text = f.read()

    # Split into chunks of N words
    words = full_text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words) - chunk_size + 1, stride)]
    return chunks

def load_uploaded_file_and_chunk(file_name, chunk_size=300, stride=100):
    # /Users/luojidong/程式/arxiv-copilot/scripts/pdf_chunks/0704.0001.txt
    path = os.path.join(upload_file_chunk_dir, f"{file_name}.txt") 
    if not os.path.exists(path):
        raise FileNotFoundError(f"uploaded file not found: {path}")
    with open(path, "r") as f:
        full_text = f.read()

    words = full_text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words) - chunk_size + 1, stride)]
    return chunks

def retrieve_top_chunks(query, chunks, top_k=10):
    if not chunks:
        raise ValueError("No chunks available for retrieval.")
    chunk_embeddings = model.encode(chunks)
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def standalone_answer(query, chunks, model_name, base64_image, api_key=None):
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
        temperature=0.3,
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

# Example
# if __name__ == "__main__":
#     arxiv_id = "0704.0001"
#     query = "How does the paper calculate diphoton production cross-sections?"
#     chunks = load_and_chunk(arxiv_id)
#     top_chunks = retrieve_top_chunks(query, chunks, top_k=5)
#     answer = standalone_answer(query, top_chunks, MODEL_NAME, api_key=os.getenv("OPENAI_API_KEY"))
#
#     print("🔎 Answer:")
#     print(answer)

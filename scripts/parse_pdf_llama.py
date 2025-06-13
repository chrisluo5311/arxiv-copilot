import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__)) # /Users/luojidong/程式/arxiv-copilot/scripts
source_path = os.path.join(current_dir, "pdfs")  # /Users/luojidong/程式/arxiv-copilot/scripts/pdfs
chunk_output_path = os.path.join(current_dir, "pdf_chunks") # /Users/luojidong/程式/arxiv-copilot/scripts/pdf_chunks
upload_file_output_path = os.path.join(current_dir, "upload_file_chunks") # /Users/luojidong/程式/arxiv-copilot/scripts/pdf_chunks

# === Load API Key from .env ===
load_dotenv()
api_key = os.getenv("LLAMAPARSE_API_KEY")
current_dir = os.path.dirname(os.path.abspath(__file__))

if not api_key:
    raise ValueError("LlamaParse API key not found in .env file!")


# === Initialize parser ===
parser = LlamaParse(api_key=api_key, verbose=True, result_type="markdown")

# === Parse PDF  ===
def parse_pdf_with_llamaparse(arxiv_id: str):
    # if not os.path.exists(chunk_output_path):
    #     os.makedirs(chunk_output_path)
    output_file_name = os.path.join(chunk_output_path, f"{arxiv_id}.txt")
    if os.path.exists(output_file_name) and os.path.getsize(output_file_name) > 0:
        return
    arxiv_pdf = os.path.join(source_path, f"{arxiv_id}.pdf")
    documents = parser.load_data(arxiv_pdf)
    with open(output_file_name, "w") as f:
        for doc in tqdm(documents, desc="Parsing PDF"):
            f.write(doc.text + "\n\n")
    print(f"✅ Parsed text saved to: {output_file_name}")

# === Parse File  ===
def parse_file_with_llamaparse(file_name: str):
    if not os.path.exists(upload_file_output_path):
        os.makedirs(upload_file_output_path)
    output_file_name = os.path.join(upload_file_output_path, f"{file_name}")
    if os.path.exists(output_file_name) and os.path.getsize(output_file_name) > 0:
        return
    upload_file = os.path.join(source_path, f"{file_name}")
    documents = parser.load_data(upload_file)
    with open(output_file_name, "w") as f:
        for doc in tqdm(documents, desc="Parsing Uploaded File..."):
            f.write(doc.text + "\n\n")
    print(f"✅ Parsed text saved to: {output_file_name}")

# === Run example ===
if __name__ == "__main__":
    arxiv_id = "0704.0001"
    parse_pdf_with_llamaparse(arxiv_id)

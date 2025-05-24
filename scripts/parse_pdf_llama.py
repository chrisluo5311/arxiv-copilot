import os
from dotenv import load_dotenv
from llama_parse import LlamaParse

# === Load API Key from .env ===
load_dotenv()
api_key = os.getenv("LLAMAPARSE_API_KEY")

if not api_key:
    raise ValueError("❌ LlamaParse API key not found in .env file!")


# === Initialize parser ===
parser = LlamaParse(api_key=api_key, verbose=True, result_type="markdown")

# === Example usage ===
def parse_pdf_with_llamaparse(arxiv_id):
    path = f"../pdfs/{arxiv_id}.pdf"
    documents = parser.load_data(path)
    output_path = f"../pdf_chunks/{arxiv_id}.txt"

    with open(output_path, "w") as f:
        for doc in documents:
            f.write(doc.text + "\n\n")

    print(f"✅ Parsed text saved to: {output_path}")

# === Run example ===
if __name__ == "__main__":
    arxiv_id = "0704.0001"  # test id
    parse_pdf_with_llamaparse(arxiv_id)

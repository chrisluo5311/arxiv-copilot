import os
from dotenv import load_dotenv
from llama_parse import LlamaParse
from tqdm import tqdm
from .storage_s3 import (
    put_text_to_s3,
    S3_CHUNK_PREFIX,
    S3_UPLOAD_CHUNK_PREFIX,
    upload_file_to_s3,
    S3_UPLOAD_PREFIX,
)

# Load API Key from .env
load_dotenv()
api_key = os.getenv("LLAMAPARSE_API_KEY")

if not api_key:
    raise ValueError("LlamaParse API key not found in .env file!")

# Initialize parser
parser = LlamaParse(api_key=api_key, verbose=True, result_type="markdown")

# use /tmp as temporary space, not take over EC2 permanent disk
TMP_DIR = "/tmp/arxiv-copilot"
os.makedirs(TMP_DIR, exist_ok=True)

def parse_pdf_with_llamaparse(arxiv_id: str, pdf_bytes: bytes | None = None):
    """
    Parse the PDF with LlamaParse
    Args:
        arxiv_id: the arXiv ID of the paper
    Returns:
        None
    """
    tmp_pdf_path = os.path.join(TMP_DIR, f"{arxiv_id}.pdf")

    # write temporary PDF file
    if pdf_bytes is not None:
        with open(tmp_pdf_path, "wb") as f:
            f.write(pdf_bytes)
    elif not os.path.exists(tmp_pdf_path):
        raise FileNotFoundError("No PDF bytes provided and temp pdf not found.")

    documents = parser.load_data(tmp_pdf_path)
    texts = []
    for doc in tqdm(documents, desc="Parsing PDF"):
        texts.append(doc.text)

    full_text = "\n\n".join(texts)
    s3_key = S3_CHUNK_PREFIX + f"{arxiv_id}.txt"
    put_text_to_s3(s3_key, full_text)
    print(f"✅ Parsed text uploaded to S3: {s3_key}")

    # clean up temporary PDF file
    try:
        os.remove(tmp_pdf_path)
    except OSError:
        pass

def parse_file_with_llamaparse(file_name: str, file_bytes: bytes):
    """
    Parse the uploaded file with LlamaParse
    Args:
        file_name: the name of the uploaded file
    Returns:
        None
    """
    tmp_path = os.path.join(TMP_DIR, file_name)
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    # backup original file to S3
    upload_file_to_s3(tmp_path, S3_UPLOAD_PREFIX + file_name)

    documents = parser.load_data(tmp_path)
    texts = []
    for doc in tqdm(documents, desc="Parsing Uploaded File..."):
        texts.append(doc.text)

    full_text = "\n\n".join(texts)
    s3_key = S3_UPLOAD_CHUNK_PREFIX + f"{file_name}.txt"
    put_text_to_s3(s3_key, full_text)
    print(f"✅ Uploaded file parsed text uploaded to S3: {s3_key}")

    try:
        os.remove(tmp_path)
    except OSError:
        pass

# arxiv-copilot

arxiv-copilot is an interactive tool for searching, downloading, parsing, and Q&A with arXiv papers. It supports PDF parsing, abstract retrieval, RAG-based Q&A, and Chatbot mode. The frontend is built with [Streamlit](https://streamlit.io/), and the backend integrates the OpenAI API and LlamaParse.

## Features

- **Fast Retrieval**: Search arXiv paper abstracts by keyword.
- **Chatbot Mode**: Interact with an AI assistant, supporting abstract search tools.
- **File Q&A Mode**: Enter an arXiv ID to download and parse the PDF, then ask questions about the paper.
- **PDF Parsing**: Automatically download and parse PDFs with LlamaParse, storing chunks in `pdf_chunks/`.
- **RAG (Retrieval-Augmented Generation)**: Generate answers based on retrieved document chunks.

## Project Structure

- `streamlit_app.py`: Main Streamlit application.
- `custom_func.py`: Custom abstract search tools.
- `scripts/`: Helper scripts (PDF download, parsing, RAG Q&A, etc.).
- `data/arxiv-metadata-oai-snapshot.json`: arXiv paper metadata.
- `pdfs/`, `pdf_chunks/`: Original PDFs and parsed text chunks.
- `embeddings/`: Vector index storage.
- `test/`: Test scripts.

## Installation

1. **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```

2. **Set environment variables**
    - Create a `.env` file and add:
      ```
      OPENAI_API_KEY=your_openai_key
      LLAMAPARSE_API_KEY=your_llamaparse_key
      ```

3. **Run the application**
    ```sh
    streamlit run streamlit_app.py
    ```

## Usage

1. After launch, enter your OpenAI API key in the sidebar.
2. Select a mode (Fast Retrieval, Chatbot, File Q&A).
3. In File Q&A mode, enter an arXiv ID and your question to automatically download, parse, and query the paper.
4. In Chatbot mode, interact with the AI assistant, which supports abstract search tools.

## Main Dependencies

- [Streamlit](https://streamlit.io/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [LlamaParse](https://github.com/run-llama/llama_parse)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
- [FAISS](https://github.com/facebookresearch/faiss) (if using vector search)

## License

This project is for academic and research use only.

---

For more details, please refer to the script files or [streamlit_app.py](streamlit_app.py).
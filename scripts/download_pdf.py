import requests
import os

def download_arxiv_pdf(arxiv_id: str) -> bytes:
    """ 
    Download the PDF from the arXiv
    Args:
        arxiv_id: the arXiv ID of the paper
    Returns:
        True if the PDF is downloaded successfully, False otherwise
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print("Downloading PDF from:", url)
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        print(f"❌ Failed to download: {url}")
        return None

import requests

def download_arxiv_pdf(arxiv_id: str, save_path: str) -> bool:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"✅ PDF downloaded: {save_path}")
        return True
    else:
        print(f"❌ Failed to download: {url}")
        return False


# if __name__ == "__main__":
#     # Example usage
#     download_arxiv_pdf("0704.0001", "../pdfs/0704.0001.pdf")
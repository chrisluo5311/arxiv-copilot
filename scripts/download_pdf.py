import requests
import os

current_dir = os.path.dirname(os.path.abspath(__file__)) # /Users/luojidong/程式/arxiv-copilot/scripts
pdfs_dir = os.path.join(current_dir, "pdfs")

def download_arxiv_pdf(arxiv_id: str) -> bool:
    save_path = os.path.join(pdfs_dir, f"{arxiv_id}.pdf")
    # check if the pdfs directory exists, if not, create it
    if not os.path.exists(pdfs_dir):
        os.makedirs(pdfs_dir)

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    print("Downloading PDF from:", url)
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
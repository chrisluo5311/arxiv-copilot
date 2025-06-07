import json

# ------------ File Paths ------------ #
original_arxiv_path = "/Users/luojidong/程式/arxiv-copilot/data/arxiv-metadata-oai-snapshot.json"
filtered_arxiv_path = "/Users/luojidong/程式/arxiv-copilot/data/filter/filtered_arxiv_cs_v2.jsonl"
SPIQA_testA_path = "/Users/luojidong/程式/arxiv-copilot/data/SPIQA/SPIQA_testA.json" # 118
SPIQA_testB_path = "/Users/luojidong/程式/arxiv-copilot/data/SPIQA/SPIQA_testB.json" # 65
SPIQA_testC_path = "/Users/luojidong/程式/arxiv-copilot/data/SPIQA/SPIQA_testC.json" # 314

# ------------ Functions ------------ #
def normalize_arxiv_id(arxiv_id):
    return arxiv_id.split('v')[0]

# Load your filtered IDs
with open(filtered_arxiv_path) as f:
    your_ids = set(normalize_arxiv_id(json.loads(line)["id"]) for line in f)

# Load SPIQA testA IDs
with open(SPIQA_testC_path) as f:
    spiqa_ids = set(normalize_arxiv_id(k) for k in json.load(f).keys())
    print(f"{SPIQA_testC_path}'s length is {len(spiqa_ids)}")


# Check overlap
matched_ids = your_ids & spiqa_ids
print(f"Matched papers: {len(matched_ids)} between filtered set and SPIQA testC")


# Match rate
# 115 between filtered set and SPIQA testA
# 65 between filtered set and SPIQA testB
# 314 between filtered set and SPIQA testC
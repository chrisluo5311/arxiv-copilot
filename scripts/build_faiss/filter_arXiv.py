import json
from tqdm import tqdm

# === Example of arXiv record structure ===
# {
#     "id": "0704.0006",
#     "submitter": "Yue Hin Pong",
#     "authors": "Y. H. Pong and C. K. Law",
#     "title": "Bosonic characters of atomic Cooper pairs across resonance",
#     "comments": "6 pages, 4 figures, accepted by PRA",
#     "journal-ref": null,
#     "doi": "10.1103/PhysRevA.75.043613",
#     "report-no": null,
#     "categories": "cond-mat.mes-hall",
#     "license": null,
#     "abstract": "  We study the two-particle wave function of paired atoms in a Fermi gas with\ntunable interaction strengths controlled by Feshbach resonance. The Cooper pair\nwave function is examined for its bosonic characters, which is quantified by\nthe correction of Bose enhancement factor associated with the creation and\nannihilation composite particle operators. An example is given for a\nthree-dimensional uniform gas. Two definitions of Cooper pair wave function are\nexamined. One of which is chosen to reflect the off-diagonal long range order\n(ODLRO). Another one corresponds to a pair projection of a BCS state. On the\nside with negative scattering length, we found that paired atoms described by\nODLRO are more bosonic than the pair projected definition. It is also found\nthat at $(k_F a)^{-1} \\ge 1$, both definitions give similar results, where more\nthan 90% of the atoms occupy the corresponding molecular condensates.\n",
#     "versions": [
#         {
#             "version": "v1",
#             "created": "Sat, 31 Mar 2007 04:24:59 GMT"
#         }
#     ],
#     "update_date": "2015-05-13",
#     "authors_parsed": [
#         [
#             "Pong",
#             "Y. H.",
#             ""
#         ],
#         [
#             "Law",
#             "C. K.",
#             ""
#         ]
#     ]
# }

# === Settings ===
input_path = "/Users/luojidong/程式/arxiv-copilot/data/arxiv-metadata-oai-snapshot.json"  # update path if needed
output_path = "/Users/luojidong/程式/arxiv-copilot/data/filter/filtered_arxiv_cs_v2.jsonl"

SPIQA_testA_path = "/Users/luojidong/程式/arxiv-copilot/data/SPIQA/SPIQA_testA.json" # 118
SPIQA_testB_path = "/Users/luojidong/程式/arxiv-copilot/data/SPIQA/SPIQA_testB.json"
SPIQA_testC_path = "/Users/luojidong/程式/arxiv-copilot/data/SPIQA/SPIQA_testC.json"
SPIQA_test_list = [SPIQA_testA_path, SPIQA_testB_path, SPIQA_testC_path]

# Filter records for CS-related papers from 2018 to 2023
cs_categories = [
    "cs.",  # match all Computer Science categories
]

def normalize_arxiv_id(arxiv_id):
    return arxiv_id.split('v')[0]

def is_cs_paper(record):
    cats = record.get("categories", "")
    return any(cat.lower().startswith("cs.") for cat in cats.split())

def is_in_year_range(record, start=2018, end=2023):
    try:
        versions = record.get("versions", [])
        if not versions:
            return False
        created_date = versions[0].get("created", "")  # e.g., "Mon, 2 Apr 2007 19:18:42 GMT"
        year = int(created_date.split()[3])
        print(year)
        return start <= year <= end
    except:
        return False


# 1. Aggregate all SPIQA IDs into a single set up front
all_spiqa_ids = set()
for spiqa_path in SPIQA_test_list:
    with open(spiqa_path, "r") as spiqa_file:
        spiqa_ids = (normalize_arxiv_id(k) for k in json.load(spiqa_file).keys())
        all_spiqa_ids.update(spiqa_ids)

# 2. Process input file efficiently
with open(input_path, "r") as fin, open(output_path, "w") as fout:
    for line in tqdm(fin, desc="Filtering CS papers matching SPIQA test set IDs"):
        record = json.loads(line)
        if is_cs_paper(record):
            arxiv_id = normalize_arxiv_id(record["id"])
            if arxiv_id in all_spiqa_ids:
                fout.write(json.dumps(record) + "\n")


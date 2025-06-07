import os
import time
import json
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import pandas as pd
from scripts.rag_generate import load_and_chunk, retrieve_top_chunks, standalone_answer
from scripts.download_pdf import download_arxiv_pdf
from scripts.parse_pdf_llama import parse_pdf_with_llamaparse
from openai import OpenAI, RateLimitError
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import base64

# === Configuration ===
output_dir = "./multimodal_CoT/"
output_CSV_path = "rag_evaluation_results.csv"
output_JSONL_path = "rag_evaluation_results.jsonl"
SPIQA_testA_path = "/Users/luojidong/程式/arxiv-copilot/data/SPIQA/SPIQA_testA.json" # 118
SPIQA_testA_image_path = "/Users/luojidong/程式/arxiv-copilot/data/SPIQA_testA_Images" # 118
SPIQA_testB_path = "/Users/luojidong/程式/arxiv-copilot/data/SPIQA/SPIQA_testB.json" # 65
SPIQA_testC_path = "/Users/luojidong/程式/arxiv-copilot/data/SPIQA/SPIQA_testC.json" # 314
TOP_K = 5
MODEL_NAME = "all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Load SPIQA Test-A Metadata ===
with open(SPIQA_testA_path, "r") as f:
    spiqa_data = json.load(f)
    print(f"Loaded {len(spiqa_data)} SPIQA entries from {SPIQA_testA_path}")

# === Load SBERT model ===
model = SentenceTransformer(MODEL_NAME)

# === Helper Functions ===
def normalize_arxiv_id(arxiv_id):
    return arxiv_id.split('v')[0]

def encode_image_to_base64(image_path):
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return ""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded

def safe_openai_chat_completion(client, **kwargs):
    max_retries = 5
    backoff = 2  # exponential backoff base in seconds
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            wait = backoff ** attempt
            print(f"⏳ Rate limit hit. Retrying in {wait:.1f}s (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    print("❌ Max retries reached.")
    return None

# === Evaluation Containers ===
results = []
for arxiv_id, content in tqdm(spiqa_data.items(), desc="Evaluating QA"):
    qas = content.get("qa", [])
    for qa in qas:
        question = qa["question"]
        reference = qa["answer"]
        reference_img_file_name = qa.get("reference", "")
        image_path = os.path.join(SPIQA_testA_image_path, f"{arxiv_id}/{reference_img_file_name}")
        # Encode reference image to base64
        base64_img = encode_image_to_base64(image_path)
        # === RAG-based Answer ===
        start_time = time.time()
        try:
            # normalize arxiv_id e.g., "1611.04684v1" to "1611.04684"
            normalized_arxiv_id = normalize_arxiv_id(arxiv_id)
            # download, parse and chunk
            # download_arxiv_pdf(normalized_arxiv_id)
            parse_pdf_with_llamaparse(normalized_arxiv_id)
            chunks = load_and_chunk(normalized_arxiv_id)
            top_chunks = retrieve_top_chunks(question, chunks, top_k=TOP_K)
            answer_rag = standalone_answer(question, top_chunks, OPENAI_MODEL, base64_img, api_key=OPENAI_API_KEY)
        except Exception as e:
            print(f"Error processing {arxiv_id}: {e}")
            answer_rag = ""
        latency_rag = time.time() - start_time

        # === Vanilla GPT Answer ===
        client = OpenAI(api_key=OPENAI_API_KEY)
        start_time = time.time()
        try:
            # Prompt GPT without any chunked context
            response = safe_openai_chat_completion(client,
                                        model=OPENAI_MODEL,
                                        temperature=0.3,
                                        messages=[
                                            {
                                                "role": "user",
                                                "content": [
                                                    {"type": "text", "text": question},
                                                    {"type": "image_url", "image_url": {
                                                        "url": f"data:image/png;base64,{base64_img}"
                                                    }}
                                                ]
                                            }
                                        ])
            if response is not None:
                answer_vanilla = response.choices[0].message.content.strip()
            else:
                answer_vanilla = ""
        except Exception as e:
            print(f"Error with Vanilla GPT for {arxiv_id}: {e}")
            answer_vanilla = ""
        latency_vanilla = time.time() - start_time

        # === Similarity-based scoring ===
        emb_ref = model.encode(reference, convert_to_tensor=True)
        emb_rag = model.encode(answer_rag, convert_to_tensor=True)
        emb_vanilla = model.encode(answer_vanilla, convert_to_tensor=True)

        score_rag = util.cos_sim(emb_ref, emb_rag).item()
        score_vanilla = util.cos_sim(emb_ref, emb_vanilla).item()

        results.append({
            "arxiv_id": arxiv_id,
            "question": question,
            "reference": reference,
            "answer_rag": answer_rag,
            "answer_vanilla": answer_vanilla,
            "similarity_rag": score_rag,
            "similarity_vanilla": score_vanilla,
            "latency_rag": latency_rag,
            "latency_vanilla": latency_vanilla
        })


df = pd.DataFrame(results)
# Save results to CSV
df.to_csv(output_dir+output_CSV_path, index=False)

# Save as JSONL
with open(output_dir+output_JSONL_path, "w") as f:
    for row in df.to_dict(orient="records"):
        f.write(json.dumps(row) + "\n")

print("✅ Results saved to 'rag_evaluation_results.csv' and 'rag_evaluation_results.jsonl'")

# --- Boxplot of similarity scores ---
plt.figure(figsize=(8, 6))
plt.boxplot([df["similarity_rag"], df["similarity_vanilla"]], tick_labels=["RAG", "Vanilla GPT"])
plt.title("Cosine Similarity Score Distribution")
plt.ylabel("Similarity to Ground Truth")
plt.grid(True)
plt.savefig(output_dir+"similarity_boxplot.png")
plt.show()

# --- Bar chart of average latency ---
avg_latency_rag = df["latency_rag"].mean()
avg_latency_vanilla = df["latency_vanilla"].mean()

plt.figure(figsize=(6, 5))
plt.bar(["RAG", "Vanilla GPT"], [avg_latency_rag, avg_latency_vanilla], color=["skyblue", "orange"])
plt.title("Average Latency (seconds)")
plt.ylabel("Time (s)")
plt.savefig(output_dir+"latency_barplot.png")
plt.show()

# --- Scatter plot of similarity vs latency ---
plt.figure(figsize=(8, 6))
plt.scatter(df["latency_rag"], df["similarity_rag"], alpha=0.6, label="RAG", color="blue")
plt.scatter(df["latency_vanilla"], df["similarity_vanilla"], alpha=0.6, label="Vanilla GPT", color="orange")
plt.title("Similarity vs. Latency")
plt.xlabel("Latency (seconds)")
plt.ylabel("Similarity Score")
plt.legend()
plt.grid(True)
plt.savefig(output_dir+"similarity_vs_latency.png")
plt.show()

# --- Calculate Precision / Recall / F1 / Accuracy
THRESHOLD = 0.7

df["correct_rag"] = df["similarity_rag"] >= THRESHOLD
df["correct_vanilla"] = df["similarity_vanilla"] >= THRESHOLD

# Ground truth: 1 for all answers (since each SPIQA question has 1 gold answer)
y_true = [1] * len(df)

# RAG metrics
y_pred_rag = df["correct_rag"].astype(int)
acc_rag = accuracy_score(y_true, y_pred_rag)
prec_rag = precision_score(y_true, y_pred_rag)
rec_rag = recall_score(y_true, y_pred_rag)
f1_rag = f1_score(y_true, y_pred_rag)
sim_rag = df["similarity_rag"].mean()

# Vanilla GPT metrics
y_pred_vanilla = df["correct_vanilla"].astype(int)
acc_van = accuracy_score(y_true, y_pred_vanilla)
prec_van = precision_score(y_true, y_pred_vanilla)
rec_van = recall_score(y_true, y_pred_vanilla)
f1_van = f1_score(y_true, y_pred_vanilla)
sim_van = df["similarity_vanilla"].mean()

# Metrics to display
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "Avg Similarity"]
rag_row = [acc_rag, prec_rag, rec_rag, f1_rag, sim_rag]
van_row = [acc_van, prec_van, rec_van, f1_van, sim_van]

# Format for table
table_data = [
    ["RAG"] + [f"{val:.2f}" for val in rag_row],
    ["Vanilla GPT"] + [f"{val:.2f}" for val in van_row]
]

# Plot table
fig, ax = plt.subplots(figsize=(9, 2.5))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=table_data, colLabels=["Model"] + metrics, loc="center", cellLoc="center")

plt.title("Multimodal CoT RAG vs Vanilla GPT Evaluation", pad=20)
plt.savefig(output_dir+"evaluation_summary_table.png", bbox_inches="tight")
plt.show()
import boto3
import os
from dotenv import load_dotenv
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
INDEX_ARN_NAME = os.getenv("INDEX_ARN_NAME")

s3vec = boto3.client("s3vectors", region_name=AWS_REGION)

def count_vectors_in_index():
    paginator = s3vec.get_paginator("list_vectors")

    total = 0
    pages = paginator.paginate(indexArn=INDEX_ARN_NAME)

    for page in pages:
        vecs = page.get("vectors", [])
        total += len(vecs)

    return total

if __name__ == "__main__":
    total = count_vectors_in_index()
    print(f"Total vectors in index: {total}") 
    # curently 55800 vectors in index

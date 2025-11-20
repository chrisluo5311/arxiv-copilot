import os
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
VECTOR_BUCKET_NAME = os.getenv("VECTOR_BUCKET_NAME")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME")
VECTOR_ARN_NAME = os.getenv("VECTOR_ARN_NAME")
INDEX_ARN_NAME = os.getenv("INDEX_ARN_NAME")

if __name__ == "__main__":
    s3vec = boto3.client("s3vectors", region_name=AWS_REGION)

    print("=== ENV ===")
    print("AWS_REGION       :", AWS_REGION)
    print("VECTOR_BUCKET_NAME:", VECTOR_BUCKET_NAME)
    print("VECTOR_INDEX_NAME :", VECTOR_INDEX_NAME)

    print("\n=== 1. vector bucket exists? ===")
    try:
        resp = s3vec.get_vector_bucket(vectorBucketName=VECTOR_BUCKET_NAME)
        bucket = resp.get("vectorBucket")
        print(f"bucket name: {bucket.get('vectorBucketName')}")
        print(f"bucket arn: {bucket.get('vectorBucketArn')}")
    except Exception as e:
        print("exists error:", e)
    
    print("\n=== 2. vector index exists? ===")
    try:
        resp = s3vec.get_index(indexArn=INDEX_ARN_NAME)
        index = resp.get("index")
        print(f"index name: {index.get('indexName')}")
        print(f"index arn: {index.get('indexArn')}")
        print(f"index metadata: {index.get('metadata')}")
    except Exception as e:
        print("exists error:", e)
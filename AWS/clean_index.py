import os
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
VECTOR_BUCKET_NAME = os.getenv("VECTOR_BUCKET_NAME")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME")
VECTOR_ARN_NAME = os.getenv("VECTOR_ARN_NAME")
INDEX_ARN_NAME = os.getenv("INDEX_ARN_NAME")

s3vec = boto3.client("s3vectors", region_name=AWS_REGION)

def clean_index():
    resp = s3vec.delete_index(indexArn=INDEX_ARN_NAME)
    print("Index deleted successfully")

if __name__ == "__main__":
    clean_index()
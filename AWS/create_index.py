import os
import boto3
from dotenv import load_dotenv
load_dotenv()
AWS_REGION = os.getenv("AWS_REGION")
VECTOR_BUCKET_NAME = os.getenv("VECTOR_BUCKET_NAME")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME")
VECTOR_ARN_NAME = os.getenv("VECTOR_ARN_NAME")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM"))

s3vec = boto3.client("s3vectors", region_name=AWS_REGION)

def create_index():
    resp = s3vec.create_index(vectorBucketArn=VECTOR_ARN_NAME, indexName=VECTOR_INDEX_NAME, dataType='float32', dimension=EMBEDDING_DIM, distanceMetric='cosine', metadataConfiguration={'nonFilterableMetadataKeys': ["paper_id", "title", "year", "categories", "authors", "abstract"]})
    indexArn = resp.get("indexArn")
    print("Index ARN:", indexArn)

if __name__ == "__main__":
    create_index()

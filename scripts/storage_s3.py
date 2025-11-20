import os
import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
S3_BUCKET = os.getenv("S3_BUCKET")

S3_CHUNK_PREFIX = os.getenv("S3_CHUNK_PREFIX")
S3_UPLOAD_CHUNK_PREFIX = os.getenv("S3_UPLOAD_CHUNK_PREFIX")
S3_UPLOAD_PREFIX = os.getenv("S3_UPLOAD_PREFIX")

s3 = boto3.client("s3", region_name=AWS_REGION)


def s3_key_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def put_text_to_s3(key: str, text: str) -> None:
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=text.encode("utf-8"))


def read_text_from_s3(key: str) -> str:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read().decode("utf-8")


def upload_file_to_s3(local_path: str, key: str) -> None:
    s3.upload_file(local_path, S3_BUCKET, key)

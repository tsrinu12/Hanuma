import os
from functools import lru_cache

import boto3
from botocore.client import Config

from .config import settings


@lru_cache(maxsize=1)
def s3_client():
    kwargs = dict(
        region_name=settings.aws_region,
        config=Config(signature_version="s3v4"),
    )
    if settings.s3_endpoint_url:
        kwargs["endpoint_url"] = settings.s3_endpoint_url
    if settings.s3_access_key:
        kwargs["aws_access_key_id"] = settings.s3_access_key
        kwargs["aws_secret_access_key"] = settings.s3_secret_key
    return boto3.client("s3", **kwargs)


def ensure_buckets():
    """Create local MinIO buckets on first boot. Skipped silently on real S3."""
    client = s3_client()
    for bucket in (settings.s3_bucket_raw, settings.s3_bucket_hls):
        try:
            client.head_bucket(Bucket=bucket)
        except Exception:
            try:
                client.create_bucket(Bucket=bucket)
            except Exception:
                pass


def upload_file(local_path: str, bucket: str, key: str, content_type: str = "application/octet-stream"):
    s3_client().upload_file(
        local_path, bucket, key,
        ExtraArgs={"ContentType": content_type, "ACL": "public-read"} if "hls" in bucket else {"ContentType": content_type},
    )


def public_url(bucket: str, key: str) -> str:
    if settings.cloudfront_domain:
        return f"https://{settings.cloudfront_domain}/{key}"
    if settings.s3_endpoint_url:
        return f"{settings.s3_endpoint_url}/{bucket}/{key}"
    return f"https://{bucket}.s3.{settings.aws_region}.amazonaws.com/{key}"

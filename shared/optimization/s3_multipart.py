"""Adaptive S3 multipart uploader.

Picks part size based on file size so we never trip the 10 000-part S3 limit
and never waste a small upload on a giant part. Uploads parts in parallel via
boto3's built-in TransferManager (which handles retries, backoff, and
in-flight throttling).

Why this matters: single-PUT uploads peak ~5GB and hold the connection open
for the whole transfer; multipart restarts only the broken part on failure,
parallelizes throughput, and is the only sane way to upload long-form video.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import Config as BotoConfig

log = logging.getLogger("opt.s3_multipart")

# S3 limits: min part size = 5 MiB (except last), max parts per upload = 10 000.
_MIN_PART = 5 * 1024 * 1024
_MAX_PARTS = 10_000


@dataclass
class UploadResult:
    bucket: str
    key: str
    bytes_uploaded: int
    part_size: int
    num_parts: int


def pick_part_size(file_size: int, target_parts: int = 1000) -> int:
    """Choose a part size so total parts ≈ target_parts but never exceed
    S3's 10 000-part cap. Rounded up to the next 1 MiB boundary."""
    if file_size <= 0:
        return _MIN_PART
    desired = max(_MIN_PART, math.ceil(file_size / target_parts))
    # Round up to MiB so logs look clean
    desired = ((desired + (1 << 20) - 1) >> 20) << 20
    # Hard floor: ensure we stay under 10 000 parts
    floor = math.ceil(file_size / _MAX_PARTS)
    return max(desired, floor, _MIN_PART)


def make_client(region: str, endpoint_url: Optional[str] = None,
                access_key: Optional[str] = None, secret_key: Optional[str] = None):
    kwargs = dict(region_name=region, config=BotoConfig(
        signature_version="s3v4",
        retries={"max_attempts": 5, "mode": "adaptive"},
        max_pool_connections=64,
    ))
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if access_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client("s3", **kwargs)


def upload_file(
    s3_client,
    local_path: str,
    bucket: str,
    key: str,
    *,
    target_parts: int = 1000,
    max_concurrency: int = 8,
    extra_args: Optional[dict] = None,
) -> UploadResult:
    """Multipart-upload `local_path` to `s3://bucket/key`."""
    size = os.path.getsize(local_path)
    part_size = pick_part_size(size, target_parts=target_parts)
    cfg = TransferConfig(
        multipart_threshold=_MIN_PART,
        multipart_chunksize=part_size,
        max_concurrency=max_concurrency,
        use_threads=True,
    )
    log.info("multipart upload size=%d part=%d concurrency=%d -> s3://%s/%s",
             size, part_size, max_concurrency, bucket, key)
    s3_client.upload_file(
        Filename=local_path, Bucket=bucket, Key=key,
        Config=cfg, ExtraArgs=extra_args or {},
    )
    return UploadResult(
        bucket=bucket, key=key, bytes_uploaded=size, part_size=part_size,
        num_parts=math.ceil(size / part_size),
    )


def upload_stream(
    s3_client, stream, bucket: str, key: str, *,
    part_size: int = _MIN_PART, extra_args: Optional[dict] = None,
) -> UploadResult:
    """Manual multipart upload from an iterable of bytes (e.g. an HTTP request
    body or a generator that yields ffmpeg output). Use this when you don't
    have the total size up-front."""
    extra_args = extra_args or {}
    mpu = s3_client.create_multipart_upload(Bucket=bucket, Key=key, **extra_args)
    upload_id = mpu["UploadId"]
    parts = []
    buf = bytearray()
    part_no = 1
    total = 0
    try:
        for chunk in stream:
            buf.extend(chunk)
            while len(buf) >= part_size:
                part = bytes(buf[:part_size])
                del buf[:part_size]
                r = s3_client.upload_part(Bucket=bucket, Key=key,
                                          PartNumber=part_no, UploadId=upload_id,
                                          Body=part)
                parts.append({"ETag": r["ETag"], "PartNumber": part_no})
                total += len(part)
                part_no += 1
        if buf:
            r = s3_client.upload_part(Bucket=bucket, Key=key,
                                      PartNumber=part_no, UploadId=upload_id,
                                      Body=bytes(buf))
            parts.append({"ETag": r["ETag"], "PartNumber": part_no})
            total += len(buf)
        s3_client.complete_multipart_upload(
            Bucket=bucket, Key=key, UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
    except Exception:
        s3_client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise

    return UploadResult(
        bucket=bucket, key=key, bytes_uploaded=total, part_size=part_size,
        num_parts=len(parts),
    )

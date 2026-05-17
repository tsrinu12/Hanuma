"""Streaming reader for S3 objects.

`boto3.get_object()["Body"].read()` loads the entire object into memory — fine
for a 1 MB thumbnail, fatal for a 4 GB video. This wraps the response body
into a chunked iterator with a default 8 MiB chunk, suitable for piping
straight into ffmpeg's stdin.
"""
from __future__ import annotations

from typing import IO, Iterator

CHUNK = 8 * 1024 * 1024


def iter_object(s3_client, bucket: str, key: str, *, chunk: int = CHUNK,
                start: int = 0, end: int = -1) -> Iterator[bytes]:
    range_header = None
    if start or end >= 0:
        range_end = "" if end < 0 else str(end)
        range_header = f"bytes={start}-{range_end}"

    kwargs = {"Bucket": bucket, "Key": key}
    if range_header:
        kwargs["Range"] = range_header
    body = s3_client.get_object(**kwargs)["Body"]
    try:
        while True:
            c = body.read(chunk)
            if not c:
                break
            yield c
    finally:
        body.close()


def copy_object_to(s3_client, bucket: str, key: str, dest: IO[bytes],
                   *, chunk: int = CHUNK) -> int:
    n = 0
    for c in iter_object(s3_client, bucket, key, chunk=chunk):
        dest.write(c)
        n += len(c)
    return n

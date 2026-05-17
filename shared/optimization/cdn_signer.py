"""Presigned URLs (S3) and signed CloudFront URLs.

S3 presigned URLs work without a CDN — useful in dev and as a fallback.
CloudFront signed URLs are what you actually use in prod: bytes ship from the
edge, not the origin, and you can lock playback to specific IP ranges or
expiry windows.
"""
from __future__ import annotations

import base64
import datetime as dt
import logging
from typing import Optional

import boto3

log = logging.getLogger("opt.cdn_signer")


def s3_presigned_get(s3_client, bucket: str, key: str, expires: int = 3600) -> str:
    return s3_client.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires,
    )


def s3_presigned_put(s3_client, bucket: str, key: str, expires: int = 3600,
                     content_type: Optional[str] = None) -> str:
    params = {"Bucket": bucket, "Key": key}
    if content_type:
        params["ContentType"] = content_type
    return s3_client.generate_presigned_url(
        "put_object", Params=params, ExpiresIn=expires,
    )


def cloudfront_signed_url(
    *,
    url: str,
    key_pair_id: str,
    private_key_pem: bytes,
    expires_in: int = 3600,
) -> str:
    """Generate a CloudFront signed URL valid for `expires_in` seconds.

    CloudFront key pair must be registered in your AWS console; pass the .pem
    bytes here. Uses the canned policy (single-URL, expiry-only) which is the
    simplest valid signature.
    """
    try:
        from botocore.signers import CloudFrontSigner  # type: ignore
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError as e:
        raise RuntimeError(
            "cloudfront_signed_url needs `cryptography` and `botocore`. "
            "pip install cryptography"
        ) from e

    private_key = serialization.load_pem_private_key(
        private_key_pem, password=None, backend=default_backend(),
    )

    def rsa_signer(message: bytes) -> bytes:
        return private_key.sign(message, padding.PKCS1v15(), hashes.SHA1())

    signer = CloudFrontSigner(key_pair_id, rsa_signer)
    expires = dt.datetime.utcnow() + dt.timedelta(seconds=expires_in)
    return signer.generate_presigned_url(url, date_less_than=expires)


def hls_origin_url(cloudfront_domain: str, hls_prefix: str, manifest: str = "master.m3u8") -> str:
    cf = cloudfront_domain.rstrip("/")
    if not cf.startswith("http"):
        cf = "https://" + cf
    return f"{cf}/{hls_prefix.strip('/')}/{manifest}"

"""distrebute optimization toolkit.

Importable from any service when the build context includes the repo root:

    from shared.optimization import (
        s3_multipart, content_dedupe, cdn_signer,        # storage / object
        lru_cache, redis_cache, bloom, stream_reader,    # memory
        batch, circuit_breaker, rate_limiter, gpu_pool,  # compute
    )

Each module is self-contained — pull in only what a service needs to keep
import-time light.
"""

__all__ = [
    "s3_multipart",
    "content_dedupe",
    "cdn_signer",
    "lru_cache",
    "redis_cache",
    "bloom",
    "stream_reader",
    "batch",
    "circuit_breaker",
    "rate_limiter",
    "gpu_pool",
]

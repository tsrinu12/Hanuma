"""Background pipeline: transcode -> upload HLS -> notify metadata + enqueue AI."""
import logging
import os
import tempfile

import httpx
import redis

from .config import settings
from .s3 import public_url, s3_client, upload_file
from .transcode import probe_duration, thumbnail, transcode_to_hls

log = logging.getLogger("video.jobs")

r = redis.Redis.from_url(settings.redis_url)


def process_video(video_id: str, raw_s3_key: str) -> None:
    log.info("Starting pipeline for video %s", video_id)

    with tempfile.TemporaryDirectory() as tmp:
        raw_local = os.path.join(tmp, "input.mp4")
        s3_client().download_file(settings.s3_bucket_raw, raw_s3_key, raw_local)

        duration = probe_duration(raw_local)
        thumb_local = os.path.join(tmp, "thumb.jpg")
        try:
            thumbnail(raw_local, thumb_local, at_seconds=min(1.0, max(duration / 2, 0.1)))
        except Exception as e:
            log.warning("thumbnail failed: %s", e)

        hls_dir = os.path.join(tmp, "hls")
        transcode_to_hls(raw_local, hls_dir)

        # Upload HLS tree to public bucket
        prefix = f"videos/{video_id}"
        for root, _, files in os.walk(hls_dir):
            for fname in files:
                local = os.path.join(root, fname)
                rel = os.path.relpath(local, hls_dir)
                key = f"{prefix}/{rel}".replace("\\", "/")
                ct = "application/vnd.apple.mpegurl" if fname.endswith(".m3u8") else \
                     "video/mp2t" if fname.endswith(".ts") else "application/octet-stream"
                upload_file(local, settings.s3_bucket_hls, key, content_type=ct)

        thumb_key = f"{prefix}/thumbnail.jpg"
        if os.path.exists(thumb_local):
            upload_file(thumb_local, settings.s3_bucket_hls, thumb_key, content_type="image/jpeg")

        master_url = public_url(settings.s3_bucket_hls, f"{prefix}/master.m3u8")
        thumb_url = public_url(settings.s3_bucket_hls, thumb_key)

        # Update metadata service
        httpx.patch(
            f"{settings.metadata_url}/videos/{video_id}",
            json={"status": "ready", "duration_seconds": duration,
                  "hls_master_url": master_url, "thumbnail_url": thumb_url},
            timeout=30.0,
        )

        # Enqueue AI work
        r.lpush("ai_jobs", f"{video_id}|{raw_s3_key}")
        log.info("Video %s ready, AI job enqueued", video_id)

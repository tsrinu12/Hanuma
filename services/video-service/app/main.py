"""Video service - upload, transcode to HLS, store on S3/CloudFront."""
import logging
import os
import shutil
import tempfile
import threading
import uuid

import httpx
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .jobs import process_video
from .s3 import ensure_buckets, s3_client

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("video")

app = FastAPI(title="distrebute.com - Video Service")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.on_event("startup")
def _startup():
    try:
        ensure_buckets()
    except Exception as e:
        log.warning("bucket setup skipped: %s", e)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/videos/upload", status_code=202)
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    description: str = Form(""),
    owner_id: str = Form(""),
):
    """Streams upload to S3, registers metadata, kicks off HLS transcode + AI pipeline."""
    if not file.filename:
        raise HTTPException(400, "missing filename")

    video_id = str(uuid.uuid4())
    raw_key = f"raw/{video_id}/{file.filename}"

    # Stream upload to S3
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        s3_client().upload_file(tmp_path, settings.s3_bucket_raw, raw_key)
    finally:
        os.unlink(tmp_path)

    # Register with metadata service
    resp = httpx.post(
        f"{settings.metadata_url}/videos",
        json={"title": title, "description": description, "owner_id": owner_id or None,
              "raw_s3_key": raw_key},
        timeout=30.0,
    )
    resp.raise_for_status()
    video = resp.json()
    httpx.patch(
        f"{settings.metadata_url}/videos/{video['id']}",
        json={"status": "processing"},
        timeout=15.0,
    )

    # Kick off async pipeline in a thread so the API returns fast.
    # In ECS/EKS, swap this for an SQS/Redis queue + dedicated worker.
    background_tasks.add_task(_run_pipeline, video["id"], raw_key)
    return {"video_id": video["id"], "status": "processing"}


def _run_pipeline(video_id: str, raw_key: str):
    try:
        process_video(video_id, raw_key)
    except Exception as e:
        log.exception("pipeline failed: %s", e)
        try:
            httpx.patch(f"{settings.metadata_url}/videos/{video_id}",
                        json={"status": "failed"}, timeout=10.0)
        except Exception:
            pass


@app.get("/videos/{video_id}/presigned")
def presigned_get(video_id: str, key: str):
    """Return a 1-hour presigned URL for the raw object (admin use)."""
    url = s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket_raw, "Key": key},
        ExpiresIn=3600,
    )
    return {"url": url}

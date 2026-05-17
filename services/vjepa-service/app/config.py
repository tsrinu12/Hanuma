"""Env-driven settings for vjepa-service."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Model: V-JEPA 2 ViT-L (frozen encoder for embeddings + retrieval)
    vjepa_model: str = os.getenv("VJEPA_MODEL", "facebook/vjepa2-vitl-fpc64-256")
    # Action-classification head fine-tuned on Something-Something V2
    vjepa_classifier_model: str = os.getenv(
        "VJEPA_CLASSIFIER_MODEL", "facebook/vjepa2-vitl-fpc64-256-ssv2"
    )
    # Number of frames sampled per clip (must match model.fpc)
    frames_per_clip: int = int(os.getenv("VJEPA_FRAMES", "64"))
    # Target spatial resolution (must match model)
    image_size: int = int(os.getenv("VJEPA_IMAGE_SIZE", "256"))
    # Device: "cuda" for GPU, "cpu" for fallback
    device: str = os.getenv("DEVICE", "cuda")
    # Embedding dimension stored in Qdrant
    embed_dim: int = int(os.getenv("VJEPA_EMBED_DIM", "1024"))

    # Vector store
    qdrant_url: str = os.getenv("QDRANT_URL", "http://qdrant:6333")
    qdrant_collection: str = os.getenv("VJEPA_COLLECTION", "vjepa_videos")

    # S3 (for fetching raw clips when given a video_id)
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    s3_endpoint_url: str = os.getenv("S3_ENDPOINT_URL", "")
    s3_access_key: str = os.getenv("S3_ACCESS_KEY", "")
    s3_secret_key: str = os.getenv("S3_SECRET_KEY", "")
    s3_bucket_raw: str = os.getenv("S3_BUCKET_RAW", "distribute-raw-uploads")

    # Upstreams
    metadata_url: str = os.getenv("METADATA_URL", "http://metadata-service:8000")


settings = Settings()

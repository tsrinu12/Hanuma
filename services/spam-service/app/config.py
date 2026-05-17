"""Settings for spam-service."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # Toxicity / abuse - unitary/toxic-bert is the standard baseline
    toxic_model: str = os.getenv("TOXIC_MODEL", "unitary/toxic-bert")
    # Multilingual hate speech (Civil Comments + extra langs)
    hate_model: str = os.getenv("HATE_MODEL", "Hate-speech-CNERG/dehatebert-mono-english")
    # Zero-shot spam classifier - useful for comment + caption spam detection
    zeroshot_model: str = os.getenv("ZEROSHOT_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    device: str = os.getenv("DEVICE", "cuda")

    # Heuristic thresholds
    max_urls_per_comment: int = int(os.getenv("MAX_URLS", "2"))
    max_caps_ratio: float = float(os.getenv("MAX_CAPS_RATIO", "0.6"))
    max_repeat_char_run: int = int(os.getenv("MAX_REPEAT_RUN", "6"))
    max_emoji_ratio: float = float(os.getenv("MAX_EMOJI_RATIO", "0.4"))
    rate_window_seconds: int = int(os.getenv("RATE_WINDOW", "60"))
    rate_max_in_window: int = int(os.getenv("RATE_MAX", "8"))

    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")


settings = Settings()

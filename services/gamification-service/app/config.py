"""Settings for gamification-service."""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://distribute:changeme@postgres:5432/distribute",
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")

    # Points awarded per action - editable via env if you want to A/B reward curves
    points_view: int = int(os.getenv("POINTS_VIEW", "1"))
    points_complete: int = int(os.getenv("POINTS_COMPLETE", "5"))
    points_like: int = int(os.getenv("POINTS_LIKE", "2"))
    points_comment: int = int(os.getenv("POINTS_COMMENT", "2"))
    points_share: int = int(os.getenv("POINTS_SHARE", "3"))
    points_upload: int = int(os.getenv("POINTS_UPLOAD", "10"))
    points_first_view_of_day: int = int(os.getenv("POINTS_FIRST_VIEW_OF_DAY", "10"))
    points_streak_bonus_per_day: int = int(os.getenv("POINTS_STREAK_BONUS", "2"))

    # Anti-fraud: minimum watched fraction to count as "view" / "complete"
    min_view_fraction: float = float(os.getenv("MIN_VIEW_FRACTION", "0.10"))
    min_complete_fraction: float = float(os.getenv("MIN_COMPLETE_FRACTION", "0.85"))

    # Unique-view dedupe window (seconds)
    dedupe_window: int = int(os.getenv("DEDUPE_WINDOW", "3600"))


settings = Settings()

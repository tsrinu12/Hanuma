"""Shared configuration primitives.

Each service has its own pydantic ``Settings`` class that inherits common
fields from here. ``MODEL_PROFILE`` is a hard switch between lite (CPU,
small models) and prod (full-size, GPU recommended).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelProfile(str, Enum):
    LITE = "lite"
    PROD = "prod"


class BaseServiceSettings(BaseSettings):
    """Common settings every service shares.

    Subclass and add service-specific fields.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=(),
    )

    service_name: str = Field(default="distrebute")
    model_profile: ModelProfile = Field(default=ModelProfile.LITE)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    @property
    def is_lite(self) -> bool:
        return self.model_profile == ModelProfile.LITE

    @property
    def is_prod(self) -> bool:
        return self.model_profile == ModelProfile.PROD

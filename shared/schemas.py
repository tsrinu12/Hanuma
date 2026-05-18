"""Common request/response shapes shared by multiple services."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    # ``model_profile`` collides with Pydantic's "model_" protected namespace.
    # Disable the warning since it's a legitimate field for us.
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(default="ok")
    service: str
    model_profile: str
    version: str = Field(default="0.1.0")


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None

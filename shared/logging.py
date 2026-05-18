"""Structured JSON logging.

All services should call ``configure_logging`` once at startup. Logs are
written to stdout as JSON so they can be picked up by a Fluent Bit / Vector /
Datadog agent in the cluster.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Mapping


class JsonFormatter(logging.Formatter):
    """Render log records as one-line JSON.

    Adds:
      - service name from ``SERVICE_NAME`` env var
      - epoch millis timestamp
      - any structured fields passed via ``extra={"fields": {...}}``
    """

    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": int(time.time() * 1000),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        fields = getattr(record, "fields", None)
        if isinstance(fields, Mapping):
            payload.update(fields)
        return json.dumps(payload, default=str)


def configure_logging(service: str | None = None, level: str | None = None) -> logging.Logger:
    """Install the JSON formatter on the root logger and return it.

    Safe to call multiple times — replaces existing handlers.
    """
    service = service or os.getenv("SERVICE_NAME", "distrebute")
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    root = logging.getLogger()
    root.setLevel(level)
    # Wipe inherited handlers (uvicorn installs its own; we want JSON only).
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter(service=service))
    root.addHandler(handler)

    for noisy in ("urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def log_event(logger: logging.Logger, msg: str, **fields: Any) -> None:
    """Emit an INFO log with structured fields attached."""
    logger.info(msg, extra={"fields": fields})

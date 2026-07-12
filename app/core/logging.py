"""Application logging configuration."""

from __future__ import annotations

import logging
import json
import sys
from typing import Optional

from .config import settings


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured application logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """Configure root logging and return the application logger."""
    log_level = (level or settings.log_level).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if settings.log_file and settings.log_file.suffix:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(settings.log_file))

    formatter: logging.Formatter
    if settings.log_json:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger("qa_knowledge")
    logger.setLevel(numeric_level)
    return logger


logger = setup_logging()

__all__ = ["logger", "setup_logging"]

"""Application logging configuration."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from .config import settings


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """Configure root logging and return the application logger."""
    log_level = (level or settings.log_level).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if settings.log_file and settings.log_file.suffix:
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(settings.log_file))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    logger = logging.getLogger("qa_knowledge")
    logger.setLevel(numeric_level)
    return logger


logger = setup_logging()

__all__ = ["logger", "setup_logging"]

"""Application configuration using Pydantic and environment variables."""

import os
import secrets
import warnings
from pathlib import Path
from typing import Optional

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    app_name: str = "Q&A Knowledge System"
    app_description: str = "Excel-powered Q&A API with semantic search"
    app_version: str = "1.0.0"
    debug: bool = False
    reload: bool = False
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    openapi_url: str = "/openapi.json"

    # API Server
    host: str = "127.0.0.1"
    port: int = 8000

    # File Paths
    knowledge_base_file: Path = (
        Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base.xlsx"
    )

    # Search Settings
    default_top_n: int = 5
    min_confidence: float = 0.0
    default_page_size: int = 100
    max_page_size: int = 500
    export_page_size: int = 500
    chart_top_n: int = 10
    knowledge_search_scan_limit: int = 2000
    knowledge_similar_scan_limit: int = 5000

    # Upload limits and validation
    max_upload_size_mb: int = 25
    allowed_upload_extensions: str = ".xlsx,.xls"

    # Embedding and model configuration
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Knowledge matching defaults
    known_threshold: float = 0.95
    variation_threshold: float = 0.80

    # Authentication (JWT)
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    log_json: bool = True

    def model_post_init(self, __context: object) -> None:
        if self.log_file is not None and not str(self.log_file).strip():
            self.log_file = None
        if not os.getenv("SECRET_KEY"):
            warnings.warn(
                "SECRET_KEY is not set in the environment; a random key was "
                "generated for this process. Existing login sessions will be "
                "invalidated on restart. Set SECRET_KEY in .env for stable "
                "sessions.",
                stacklevel=2,
            )

    # CORS
    cors_origins: list = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",  # Allow extra fields from .env
    )

    @property
    def data_path(self) -> Path:
        """Get the data directory path."""
        return self.knowledge_base_file.parent

    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum upload size in bytes."""
        return int(self.max_upload_size_mb) * 1024 * 1024

    @property
    def allowed_upload_exts(self) -> set[str]:
        """Allowed upload extensions as a normalized set."""
        return {
            x.strip().lower()
            for x in str(self.allowed_upload_extensions).split(",")
            if x.strip()
        }


# Global settings instance
settings = Settings()

__all__ = ["Settings", "settings"]

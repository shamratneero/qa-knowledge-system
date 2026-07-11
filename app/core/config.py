"""Application configuration using Pydantic and environment variables."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    app_name: str = "Q&A Knowledge System"
    app_description: str = "Excel-powered Q&A API with semantic search"
    app_version: str = "1.0.0"
    debug: bool = False
    reload: bool = False

    # API Server
    host: str = "127.0.0.1"
    port: int = 8000

    # File Paths
    knowledge_base_file: Path = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base.xlsx"
    
    # Search Settings
    default_top_n: int = 5
    min_confidence: float = 0.0
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    def model_post_init(self, __context: object) -> None:
        if self.log_file is not None and not str(self.log_file).strip():
            self.log_file = None

    # CORS
    cors_origins: list = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: list = ["*"]
    cors_allow_headers: list = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from .env

    @property
    def data_path(self) -> Path:
        """Get the data directory path."""
        return self.knowledge_base_file.parent


# Global settings instance
settings = Settings()

__all__ = ["Settings", "settings"]

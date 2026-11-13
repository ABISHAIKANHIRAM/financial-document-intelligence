"""Centralized configuration and logging setup for Financial Document Intelligence."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_HOSTNAMES = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw.strip())


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip()


def _resolve_path(path_value: str) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


class Settings(BaseModel):
    """Application settings loaded from environment variables and `.env`."""

    # Local LLM (Ollama)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:latest")
    ollama_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    ollama_num_ctx: int = Field(default=8192, ge=512, le=131072)
    ollama_timeout_seconds: int = Field(default=120, ge=5, le=1800)
    allow_non_local_ollama: bool = Field(default=False)

    # Chunking & Processing
    chunk_size: int = Field(default=3500, ge=500, le=32000)
    chunk_overlap: int = Field(default=300, ge=0, le=4000)
    max_chunks_per_document: int = Field(default=50, ge=1, le=5000)

    # OCR & PDF Rendering
    ocr_enabled: bool = Field(default=True)
    ocr_language: str = Field(default="eng")
    ocr_dpi: int = Field(default=300, ge=72, le=600)
    min_text_chars_per_page: int = Field(default=40, ge=0, le=1000)
    tesseract_cmd: Optional[str] = Field(default=None)
    poppler_path: Optional[str] = Field(default=None)

    # Storage & Logging
    database_path: Path = Field(default=PROJECT_ROOT / "data" / "financial_intel.db")
    upload_dir: Path = Field(default=PROJECT_ROOT / "uploads")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs")
    log_level: str = Field(default="INFO")

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        parsed = urlparse(cleaned)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(
                f"Invalid OLLAMA_BASE_URL '{value}'. Expected http(s)://host:port."
            )
        return cleaned

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in valid_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL '{value}'. Choose from {sorted(valid_levels)}."
            )
        return normalized

    def ensure_local_ollama(self) -> None:
        """Raise ValueError if Ollama URL points to a non-local host unless explicitly allowed."""
        if self.allow_non_local_ollama:
            return
        hostname = (urlparse(self.ollama_base_url).hostname or "").lower()
        if hostname not in LOCAL_HOSTNAMES:
            raise ValueError(
                f"Refusing non-local Ollama endpoint '{self.ollama_base_url}'. "
                "Financial documents must stay on the local machine. "
                "Use http://localhost:11434 or http://127.0.0.1:11434."
            )

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not exist."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, env_file: Optional[Path] = None) -> "Settings":
        """Load settings from `.env` (if present) and environment variables."""
        target_env = env_file or (PROJECT_ROOT / ".env")
        if target_env.exists():
            load_dotenv(dotenv_path=target_env, override=False)

        tesseract_cmd = _env_str("TESSERACT_CMD", "") or None
        poppler_path = _env_str("POPPLER_PATH", "") or None

        settings = cls(
            ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=_env_str("OLLAMA_MODEL", "qwen2.5:latest"),
            ollama_temperature=_env_float("OLLAMA_TEMPERATURE", 0.0),
            ollama_num_ctx=_env_int("OLLAMA_NUM_CTX", 8192),
            ollama_timeout_seconds=_env_int("OLLAMA_TIMEOUT_SECONDS", 120),
            allow_non_local_ollama=_env_bool("ALLOW_NON_LOCAL_OLLAMA", False),
            chunk_size=_env_int("CHUNK_SIZE", 3500),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 300),
            max_chunks_per_document=_env_int("MAX_CHUNKS_PER_DOCUMENT", 50),
            ocr_enabled=_env_bool("OCR_ENABLED", True),
            ocr_language=_env_str("OCR_LANGUAGE", "eng") or "eng",
            ocr_dpi=_env_int("OCR_DPI", 300),
            min_text_chars_per_page=_env_int("MIN_TEXT_CHARS_PER_PAGE", 40),
            tesseract_cmd=tesseract_cmd,
            poppler_path=poppler_path,
            database_path=_resolve_path(_env_str("DATABASE_PATH", "data/financial_intel.db")),
            upload_dir=_resolve_path(_env_str("UPLOAD_DIR", "uploads")),
            log_dir=_resolve_path(_env_str("LOG_DIR", "logs")),
            log_level=_env_str("LOG_LEVEL", "INFO"),
        )
        settings.ensure_local_ollama()
        if settings.chunk_overlap >= settings.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings and ensure runtime directories exist."""
    settings = Settings.from_env()
    settings.ensure_directories()
    return settings


def configure_logging(settings: Optional[Settings] = None) -> logging.Logger:
    """Configure root console and rotating file logging for the application."""
    cfg = settings or get_settings()
    cfg.ensure_directories()

    logger = logging.getLogger("financial_intel")
    logger.setLevel(getattr(logging, cfg.log_level, logging.INFO))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file = cfg.log_dir / "financial_intel.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger

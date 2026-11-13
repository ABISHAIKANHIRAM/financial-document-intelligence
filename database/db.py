"""SQLite connection, schema migration, and maintenance manager for financial document intelligence."""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from config import Settings, get_settings

logger = logging.getLogger("financial_intel.database")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    page_count INTEGER NOT NULL,
    company_name TEXT,
    country TEXT,
    industry TEXT,
    financial_year INTEGER,
    reporting_period TEXT,
    currency TEXT,
    model_used TEXT,
    inspection_json TEXT,
    extraction_json TEXT,
    validation_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    metric TEXT NOT NULL,
    value REAL,
    unit TEXT,
    year INTEGER,
    reporting_period TEXT,
    evidence TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    verification_status TEXT NOT NULL,
    flag_reasons_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'INFO',
    details_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_company ON documents(company_name);
CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(financial_year);
CREATE INDEX IF NOT EXISTS idx_metrics_document_id ON metrics(document_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric);
CREATE INDEX IF NOT EXISTS idx_metrics_year ON metrics(year);
CREATE INDEX IF NOT EXISTS idx_metrics_verification_status ON metrics(verification_status);
CREATE INDEX IF NOT EXISTS idx_audit_logs_doc_id ON audit_logs(document_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at DESC);
"""


class DatabaseManager:
    """Manages SQLite database initialization, configuration, maintenance, and transactions."""

    def __init__(self, db_path: Optional[Path | str] = None, settings: Optional[Settings] = None) -> None:
        cfg = settings or get_settings()
        self.db_path = Path(db_path or cfg.database_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Create and configure a thread-safe SQLite connection with foreign keys and WAL mode."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    @contextmanager
    def session(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager providing transactional connection management."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Initialize SQLite tables and indexes."""
        with self.session() as conn:
            conn.executescript(SCHEMA_SQL)
        logger.debug("Database initialized at %s", self.db_path)

    def vacuum(self) -> None:
        """Defragment database and reclaim unused disk space."""
        conn = self.get_connection()
        try:
            conn.execute("VACUUM;")
            logger.info("Database vacuumed successfully at %s", self.db_path)
        finally:
            conn.close()

    def get_database_stats(self) -> Dict[str, Any]:
        """Retrieve storage metrics and file size statistics."""
        size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        with self.session() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM metrics")
            metric_count = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(*) FROM audit_logs")
            log_count = cursor.fetchone()[0] or 0

        return {
            "database_path": str(self.db_path),
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 2),
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "document_count": doc_count,
            "metric_count": metric_count,
            "audit_log_count": log_count,
        }

"""Repository layer for persisting and querying documents, metrics, audit logs, and analytics."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.document_loader import ProcessedDocument
from database.db import DatabaseManager
from schemas.financial_schema import ExtractedMetric, FinancialExtractionResult
from validation.financial_validator import ValidationSummary

logger = logging.getLogger("financial_intel.repository")


class DocumentRepository:
    """Repository handling CRUD, search, audit logging, and cross-document analytics."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None) -> None:
        self.db = db_manager or DatabaseManager()

    # --------------------------------------------------------------------------
    # Audit Logging
    # --------------------------------------------------------------------------

    def log_pipeline_event(
        self,
        event_type: str,
        message: str,
        document_id: Optional[int] = None,
        level: str = "INFO",
        details: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Write a structured event to the audit_logs table."""
        now_iso = datetime.now(timezone.utc).isoformat()
        details_json = json.dumps(details, default=str) if details else None

        with self.db.session() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO audit_logs (document_id, event_type, message, level, details_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (document_id, event_type, message, level.upper(), details_json, now_iso),
            )
            log_id = cursor.lastrowid
            return log_id or 0

    def get_audit_logs(
        self,
        document_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent audit logs optionally filtered by document_id."""
        with self.db.session() as conn:
            cursor = conn.cursor()
            if document_id is not None:
                cursor.execute(
                    """
                    SELECT a.*, d.filename FROM audit_logs a
                    LEFT JOIN documents d ON a.document_id = d.id
                    WHERE a.document_id = ?
                    ORDER BY a.created_at DESC
                    LIMIT ?
                    """,
                    (document_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT a.*, d.filename FROM audit_logs a
                    LEFT JOIN documents d ON a.document_id = d.id
                    ORDER BY a.created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            rows = cursor.fetchall()
            results: List[Dict[str, Any]] = []
            for r in rows:
                item = dict(r)
                if item.get("details_json"):
                    try:
                        item["details"] = json.loads(item["details_json"])
                    except Exception:
                        item["details"] = {}
                else:
                    item["details"] = {}
                results.append(item)
            return results

    # --------------------------------------------------------------------------
    # Document Persistence & Retrieval
    # --------------------------------------------------------------------------

    def save_extraction_pipeline_output(
        self,
        document: ProcessedDocument,
        extraction_result: FinancialExtractionResult,
        validation_summary: Optional[ValidationSummary] = None,
    ) -> int:
        """Persist full pipeline output into SQLite and write initial audit trail log."""
        now_iso = datetime.now(timezone.utc).isoformat()
        doc_json = json.dumps(document.to_dict(), default=str)
        ext_json = json.dumps(extraction_result.model_dump(), default=str)
        val_json = (
            json.dumps(validation_summary.to_dict(), default=str)
            if validation_summary
            else "{}"
        )

        with self.db.session() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO documents (
                    filename, file_type, file_size_bytes, page_count,
                    company_name, country, industry, financial_year,
                    reporting_period, currency, model_used,
                    inspection_json, extraction_json, validation_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.file_name,
                    document.file_type,
                    document.file_size_bytes,
                    document.page_count,
                    extraction_result.company_name,
                    extraction_result.country,
                    extraction_result.industry,
                    extraction_result.financial_year,
                    extraction_result.reporting_period,
                    extraction_result.currency,
                    extraction_result.model_used,
                    doc_json,
                    ext_json,
                    val_json,
                    now_iso,
                ),
            )
            doc_id = cursor.lastrowid
            if doc_id is None:
                raise RuntimeError("Failed to insert document record.")

            # Insert metrics
            for m in extraction_result.metrics:
                cursor.execute(
                    """
                    INSERT INTO metrics (
                        document_id, metric, value, unit, year,
                        reporting_period, evidence, page_number,
                        verification_status, flag_reasons_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        m.metric,
                        m.value,
                        m.unit,
                        m.year,
                        m.reporting_period,
                        m.evidence,
                        m.source.page,
                        m.verification_status,
                        json.dumps(m.flag_reasons),
                        now_iso,
                    ),
                )

        # Record audit log event
        self.log_pipeline_event(
            event_type="document_persisted",
            message=f"Document '{document.file_name}' saved with {len(extraction_result.metrics)} metrics.",
            document_id=doc_id,
            level="INFO",
            details={
                "company": extraction_result.company_name,
                "year": extraction_result.financial_year,
                "verified": sum(1 for m in extraction_result.metrics if m.verification_status == "verified"),
                "flagged": sum(1 for m in extraction_result.metrics if m.verification_status == "flagged"),
            },
        )

        logger.info(
            "Saved document ID %d ('%s') with %d metrics to database",
            doc_id,
            document.file_name,
            len(extraction_result.metrics),
        )
        return doc_id

    def list_documents(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent documents with aggregated metric metrics."""
        return self.search_documents(limit=limit)

    def search_documents(
        self,
        query: str = "",
        year: Optional[int] = None,
        status_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Search and filter documents by company, filename, year, or review status."""
        conditions: List[str] = []
        params: List[Any] = []

        if query and query.strip():
            clean_q = f"%{query.strip().lower()}%"
            conditions.append("(LOWER(d.filename) LIKE ? OR LOWER(d.company_name) LIKE ?)")
            params.extend([clean_q, clean_q])

        if year:
            conditions.append("d.financial_year = ?")
            params.append(year)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        having_clause = ""
        if status_filter == "verified_only":
            having_clause = "HAVING flagged_count = 0 AND metric_count > 0"
        elif status_filter == "has_flags":
            having_clause = "HAVING flagged_count > 0"

        sql = f"""
            SELECT 
                d.id, d.filename, d.file_type, d.file_size_bytes, d.page_count,
                d.company_name, d.financial_year, d.currency, d.model_used, d.created_at,
                COUNT(m.id) as metric_count,
                SUM(CASE WHEN m.verification_status = 'verified' THEN 1 ELSE 0 END) as verified_count,
                SUM(CASE WHEN m.verification_status = 'flagged' THEN 1 ELSE 0 END) as flagged_count,
                SUM(CASE WHEN m.verification_status = 'pending' THEN 1 ELSE 0 END) as pending_count
            FROM documents d
            LEFT JOIN metrics m ON d.id = m.document_id
            {where_clause}
            GROUP BY d.id
            {having_clause}
            ORDER BY d.created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self.db.session() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, tuple(params))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve full document record including deserialized extraction and validation JSON."""
        with self.db.session() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            try:
                res["extraction_data"] = json.loads(res.get("extraction_json") or "{}")
            except Exception:
                res["extraction_data"] = {}
            try:
                res["validation_data"] = json.loads(res.get("validation_json") or "{}")
            except Exception:
                res["validation_data"] = {}
            return res

    def get_metrics_for_document(self, doc_id: int) -> List[Dict[str, Any]]:
        """Retrieve all extracted metrics for a document ordered by page number."""
        with self.db.session() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM metrics WHERE document_id = ? ORDER BY page_number ASC, id ASC
                """,
                (doc_id,),
            )
            rows = cursor.fetchall()
            results: List[Dict[str, Any]] = []
            for r in rows:
                item = dict(r)
                try:
                    item["flag_reasons"] = json.loads(item.get("flag_reasons_json") or "[]")
                except Exception:
                    item["flag_reasons"] = []
                results.append(item)
            return results

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document record, cascading metrics, and log audit event."""
        doc_record = self.get_document(doc_id)
        filename = doc_record["filename"] if doc_record else str(doc_id)

        with self.db.session() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            deleted = cursor.rowcount > 0

        if deleted:
            self.log_pipeline_event(
                event_type="document_deleted",
                message=f"Document ID {doc_id} ('{filename}') was deleted from database.",
                document_id=None,
                level="WARNING",
            )
            logger.info("Deleted document ID %d from database", doc_id)
        return deleted

    def clear_all_records(self) -> bool:
        """Clear all stored documents and metrics (used for clean resets)."""
        with self.db.session() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents;")
            cursor.execute("DELETE FROM audit_logs;")
        self.db.vacuum()
        logger.warning("All document and audit records cleared from database.")
        return True

    # --------------------------------------------------------------------------
    # Cross-Document Analytics & Time-Series Comparison
    # --------------------------------------------------------------------------

    def get_distinct_metric_names(self) -> List[str]:
        """Return sorted list of distinct metric names available across all ingested documents."""
        with self.db.session() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT metric FROM metrics ORDER BY metric ASC")
            rows = cursor.fetchall()
            return [r[0] for r in rows if r[0]]

    def get_historical_metrics_by_name(self, metric_name: str) -> List[Dict[str, Any]]:
        """Retrieve all extractions for a given metric across documents (for historical trend charts)."""
        clean_name = metric_name.strip().lower()
        with self.db.session() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    m.id, m.metric, m.value, m.unit, m.year, m.reporting_period,
                    m.verification_status, m.page_number, m.evidence,
                    d.id as doc_id, d.filename, d.company_name
                FROM metrics m
                JOIN documents d ON m.document_id = d.id
                WHERE LOWER(m.metric) = ? AND m.value IS NOT NULL
                ORDER BY m.year ASC, d.created_at ASC
                """,
                (clean_name,),
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_kpis(self) -> Dict[str, int]:
        """Aggregate high-level system KPIs across all ingested documents."""
        with self.db.session() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0] or 0

            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_metrics,
                    SUM(CASE WHEN verification_status = 'verified' THEN 1 ELSE 0 END) as verified_metrics,
                    SUM(CASE WHEN verification_status = 'flagged' THEN 1 ELSE 0 END) as flagged_metrics,
                    SUM(CASE WHEN verification_status = 'pending' THEN 1 ELSE 0 END) as pending_metrics
                FROM metrics
                """
            )
            row = cursor.fetchone()
            total_metrics = row[0] or 0
            verified_metrics = row[1] or 0
            flagged_metrics = row[2] or 0
            pending_metrics = row[3] or 0

            return {
                "total_documents": total_docs,
                "total_metrics": total_metrics,
                "verified_metrics": verified_metrics,
                "flagged_metrics": flagged_metrics,
                "pending_metrics": pending_metrics,
            }

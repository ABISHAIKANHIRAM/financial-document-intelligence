"""Export manager for generating formatted JSON, CSV, and multi-sheet Excel outputs."""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from schemas.financial_schema import ExtractedMetric, FinancialExtractionResult
from validation.financial_validator import ValidationSummary

logger = logging.getLogger("financial_intel.exports")


class ExportManager:
    """Handles formatted exports of extracted financial intelligence across JSON, CSV, and Excel."""

    @staticmethod
    def to_json_string(result: FinancialExtractionResult, indent: int = 2) -> str:
        """Serialize extraction result to formatted JSON string."""
        data = result.model_dump()
        return json.dumps(data, indent=indent, default=str)

    @classmethod
    def to_json_bytes(cls, result: FinancialExtractionResult, indent: int = 2) -> bytes:
        """Serialize extraction result to UTF-8 encoded JSON bytes for download."""
        return cls.to_json_string(result, indent=indent).encode("utf-8")

    @staticmethod
    def to_dataframe(result: FinancialExtractionResult) -> pd.DataFrame:
        """Convert extracted metrics to a structured pandas DataFrame."""
        rows: List[Dict[str, Any]] = []
        for m in result.metrics:
            rows.append(
                {
                    "Metric": m.metric,
                    "Value": m.value,
                    "Unit": m.unit or "",
                    "Year": m.year or "",
                    "Period": m.reporting_period or "",
                    "Status": m.verification_status,
                    "Page": m.source.page,
                    "Evidence": m.evidence,
                    "Document": m.source.document,
                    "Source Type": m.source.source_type or "",
                    "Flag Reasons": "; ".join(m.flag_reasons) if m.flag_reasons else "",
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "Metric",
                    "Value",
                    "Unit",
                    "Year",
                    "Period",
                    "Status",
                    "Page",
                    "Evidence",
                    "Document",
                    "Source Type",
                    "Flag Reasons",
                ]
            )
        return pd.DataFrame(rows)

    @classmethod
    def to_csv_string(cls, result: FinancialExtractionResult) -> str:
        """Convert extracted metrics to CSV text."""
        df = cls.to_dataframe(result)
        return df.to_csv(index=False)

    @classmethod
    def to_csv_bytes(cls, result: FinancialExtractionResult) -> bytes:
        """Convert extracted metrics to UTF-8 encoded CSV bytes for download."""
        return cls.to_csv_string(result).encode("utf-8")

    @classmethod
    def to_excel_bytes(
        cls,
        result: FinancialExtractionResult,
        validation_summary: Optional[ValidationSummary] = None,
    ) -> bytes:
        """Generate a professional multi-sheet Excel workbook in memory."""
        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            # Sheet 1: Company & Extraction Summary
            summary_info = [
                {"Attribute": "Document Name", "Value": result.document_name},
                {"Attribute": "Company Name", "Value": result.company_name or "N/A"},
                {"Attribute": "Country", "Value": result.country or "N/A"},
                {"Attribute": "Industry", "Value": result.industry or "N/A"},
                {"Attribute": "Financial Year", "Value": result.financial_year or "N/A"},
                {"Attribute": "Reporting Period", "Value": result.reporting_period or "N/A"},
                {"Attribute": "Reporting Currency", "Value": result.currency or "N/A"},
                {"Attribute": "Model Used", "Value": result.model_used or "N/A"},
                {"Attribute": "Total Metrics Extracted", "Value": len(result.metrics)},
                {
                    "Attribute": "Verified Metrics",
                    "Value": sum(1 for m in result.metrics if m.verification_status == "verified"),
                },
                {
                    "Attribute": "Flagged Metrics",
                    "Value": sum(1 for m in result.metrics if m.verification_status == "flagged"),
                },
                {"Attribute": "Extraction Date", "Value": result.created_at},
            ]
            df_summary = pd.DataFrame(summary_info)
            df_summary.to_excel(writer, sheet_name="Overview", index=False)

            # Sheet 2: Extracted Financial Metrics
            df_metrics = cls.to_dataframe(result)
            df_metrics.to_excel(writer, sheet_name="Financial Metrics", index=False)

            # Sheet 3: Validation & Quality Audit
            audit_rows: List[Dict[str, Any]] = []
            for m in result.metrics:
                if m.flag_reasons:
                    for reason in m.flag_reasons:
                        audit_rows.append(
                            {
                                "Metric": m.metric,
                                "Page": m.source.page,
                                "Status": m.verification_status,
                                "Issue Description": reason,
                                "Reported Value": m.value,
                                "Evidence Text": m.evidence,
                            }
                        )
            if validation_summary and validation_summary.contradictions:
                for c in validation_summary.contradictions:
                    audit_rows.append(
                        {
                            "Metric": c.get("metric"),
                            "Page": "Multi-page",
                            "Status": "contradiction",
                            "Issue Description": f"Conflicting values: {c.get('occurrences')}",
                            "Reported Value": "N/A",
                            "Evidence Text": "See individual pages",
                        }
                    )

            df_audit = (
                pd.DataFrame(audit_rows)
                if audit_rows
                else pd.DataFrame(
                    [{"Audit Status": "All extracted metrics passed deterministic validation cleanly."}]
                )
            )
            df_audit.to_excel(writer, sheet_name="Validation Audit", index=False)

        buffer.seek(0)
        return buffer.getvalue()

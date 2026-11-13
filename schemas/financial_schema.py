"""Pydantic schemas for structured financial information extraction and provenance."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SourceReference(BaseModel):
    """Detailed source provenance for an extracted financial metric."""

    model_config = ConfigDict(extra="ignore")

    page: int = Field(
        ...,
        description="1-indexed document page number where the evidence appears.",
        ge=1,
    )
    document: str = Field(
        ...,
        description="Original document filename.",
        min_length=1,
    )
    source_type: Optional[str] = Field(
        default="text_layer",
        description="Origin of the underlying text (e.g., text_layer, ocr, table, docx, xlsx, csv).",
    )


class ExtractedMetric(BaseModel):
    """A single structured financial metric extracted from a document."""

    model_config = ConfigDict(extra="ignore")

    metric: str = Field(
        ...,
        description="Standard financial metric name (e.g., Revenue, EBITDA, EBIT, Net Income).",
        min_length=1,
    )
    value: Optional[float] = Field(
        default=None,
        description="Parsed absolute numeric value. Null if unavailable.",
    )
    unit: Optional[str] = Field(
        default=None,
        description="Currency code or measurement unit (e.g., EUR, USD, %, count).",
    )
    year: Optional[int] = Field(
        default=None,
        description="Fiscal year associated with the metric (e.g., 2024).",
    )
    reporting_period: Optional[str] = Field(
        default=None,
        description="Reporting timeframe (e.g., FY 2024, Q3, H1, Full Year).",
    )
    evidence: str = Field(
        ...,
        description="Exact verbatim excerpt from the document providing proof of this metric.",
        min_length=1,
    )
    source: SourceReference = Field(
        ...,
        description="Original page and document coordinates for source traceability.",
    )
    extraction_method: str = Field(
        default="llm_extraction",
        description="Extraction technique employed (e.g., llm_extraction, table_rule).",
    )
    verification_status: Literal["pending", "verified", "flagged"] = Field(
        default="pending",
        description="Validation review status.",
    )
    flag_reasons: List[str] = Field(
        default_factory=list,
        description="Specific deterministic validation rule failure descriptions if flagged.",
    )

    @field_validator("metric")
    @classmethod
    def clean_metric_name(cls, v: str) -> str:
        """Strip whitespace and normalize standard metric casing."""
        cleaned = v.strip()
        return cleaned

    @field_validator("unit")
    @classmethod
    def normalize_unit(cls, v: Optional[str]) -> Optional[str]:
        """Normalize common currency codes and symbols."""
        if not v:
            return None
        cleaned = v.strip().upper()
        symbol_map = {
            "€": "EUR",
            "$": "USD",
            "£": "GBP",
            "¥": "JPY",
            "CHF": "CHF",
        }
        return symbol_map.get(cleaned, cleaned)


class ChunkExtractionResponse(BaseModel):
    """Raw JSON schema expected from local LLM for each document chunk."""

    model_config = ConfigDict(extra="ignore")

    company_name: Optional[str] = Field(default=None)
    country: Optional[str] = Field(default=None)
    industry: Optional[str] = Field(default=None)
    reporting_period: Optional[str] = Field(default=None)
    financial_year: Optional[int] = Field(default=None)
    currency: Optional[str] = Field(default=None)
    metrics: List[ExtractedMetric] = Field(default_factory=list)


class FinancialExtractionResult(BaseModel):
    """Complete document-level financial extraction result."""

    model_config = ConfigDict(extra="ignore")

    document_name: str = Field(..., description="Document filename.")
    company_name: Optional[str] = Field(
        default=None,
        description="Name of the reporting corporate entity.",
    )
    country: Optional[str] = Field(
        default=None,
        description="Country of incorporation or headquarters.",
    )
    industry: Optional[str] = Field(
        default=None,
        description="Operating industry or sector.",
    )
    reporting_period: Optional[str] = Field(
        default=None,
        description="Primary reporting period (e.g. FY 2024).",
    )
    financial_year: Optional[int] = Field(
        default=None,
        description="Primary financial year (e.g. 2024).",
    )
    currency: Optional[str] = Field(
        default=None,
        description="Primary reporting currency (e.g., EUR, USD).",
    )
    metrics: List[ExtractedMetric] = Field(
        default_factory=list,
        description="List of verified and traceable financial metrics.",
    )
    model_used: Optional[str] = Field(
        default=None,
        description="Exact local Ollama model used for extraction.",
    )
    chunks_processed: int = Field(default=0)
    total_chunks: int = Field(default=0)
    extraction_warnings: List[str] = Field(default_factory=list)
    extraction_errors: List[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert result to high-level summary dictionary."""
        return {
            "document_name": self.document_name,
            "company_name": self.company_name,
            "financial_year": self.financial_year,
            "currency": self.currency,
            "total_metrics": len(self.metrics),
            "verified_metrics": sum(
                1 for m in self.metrics if m.verification_status == "verified"
            ),
            "flagged_metrics": sum(
                1 for m in self.metrics if m.verification_status == "flagged"
            ),
            "pending_metrics": sum(
                1 for m in self.metrics if m.verification_status == "pending"
            ),
            "model_used": self.model_used,
            "created_at": self.created_at,
        }

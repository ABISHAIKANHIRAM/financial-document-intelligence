"""Unit tests for Pydantic financial schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from schemas.financial_schema import (
    ChunkExtractionResponse,
    ExtractedMetric,
    FinancialExtractionResult,
    SourceReference,
)


def test_source_reference_valid() -> None:
    src = SourceReference(page=3, document="annual_report_2024.pdf", source_type="text_layer")
    assert src.page == 3
    assert src.document == "annual_report_2024.pdf"
    assert src.source_type == "text_layer"


def test_source_reference_invalid_page() -> None:
    with pytest.raises(ValidationError):
        SourceReference(page=0, document="report.pdf")


def test_extracted_metric_normalizes_currency_units() -> None:
    metric = ExtractedMetric(
        metric="Revenue",
        value=12500000.0,
        unit="€",
        year=2024,
        evidence="Revenue increased to EUR 12.5 million.",
        source=SourceReference(page=1, document="report.pdf"),
    )
    assert metric.unit == "EUR"
    assert metric.value == 12500000.0
    assert metric.verification_status == "pending"


def test_extracted_metric_preserves_negative_values() -> None:
    metric = ExtractedMetric(
        metric="Net Income",
        value=-4200000.0,
        unit="USD",
        year=2024,
        evidence="Net loss for the period was USD 4.2 million.",
        source=SourceReference(page=5, document="report.pdf"),
    )
    assert metric.value == -4200000.0


def test_chunk_extraction_response_handles_nulls() -> None:
    raw_payload = {
        "company_name": "Example Corp",
        "financial_year": 2024,
        "currency": "EUR",
        "metrics": [],
    }
    resp = ChunkExtractionResponse.model_validate(raw_payload)
    assert resp.company_name == "Example Corp"
    assert resp.country is None
    assert resp.metrics == []


def test_financial_extraction_result_summary() -> None:
    res = FinancialExtractionResult(
        document_name="doc.pdf",
        company_name="Acme Corp",
        financial_year=2024,
        currency="USD",
        metrics=[
            ExtractedMetric(
                metric="Revenue",
                value=50000000.0,
                unit="USD",
                year=2024,
                evidence="Revenue of $50M",
                source=SourceReference(page=1, document="doc.pdf"),
                verification_status="verified",
            ),
            ExtractedMetric(
                metric="EBITDA",
                value=10000000.0,
                unit="USD",
                year=2024,
                evidence="EBITDA $10M",
                source=SourceReference(page=1, document="doc.pdf"),
                verification_status="flagged",
            ),
        ],
    )
    summary = res.to_summary_dict()
    assert summary["total_metrics"] == 2
    assert summary["verified_metrics"] == 1
    assert summary["flagged_metrics"] == 1
    assert summary["pending_metrics"] == 0

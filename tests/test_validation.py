"""Comprehensive unit tests for Phase 4 deterministic financial validation and source traceability."""

from __future__ import annotations

import pytest

from core.document_loader import PageContent, ProcessedDocument
from schemas.financial_schema import ExtractedMetric, FinancialExtractionResult, SourceReference
from validation.financial_validator import FinancialValidator
from validation.schema_validator import SchemaValidator


# ==============================================================================
# Schema Validator Tests
# ==============================================================================


def test_schema_validator_rejects_negative_employees() -> None:
    validator = SchemaValidator()
    metric = ExtractedMetric(
        metric="Number of Employees",
        value=-50.0,
        unit="count",
        year=2024,
        evidence="The company had 50 employees.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    report = validator.validate_metric(metric)
    assert report.is_valid is False
    assert any("headcount cannot be negative" in e.message for e in report.errors)


def test_schema_validator_rejects_unrealistic_year() -> None:
    validator = SchemaValidator()
    metric = ExtractedMetric(
        metric="Revenue",
        value=1000000.0,
        unit="EUR",
        year=1850,
        evidence="Revenue was EUR 1 million.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    report = validator.validate_metric(metric)
    assert report.is_valid is False
    assert any("Financial year" in e.message for e in report.errors)


def test_schema_validator_accepts_clean_metric() -> None:
    validator = SchemaValidator()
    metric = ExtractedMetric(
        metric="Revenue",
        value=15000000.0,
        unit="EUR",
        year=2024,
        evidence="Revenue reached EUR 15 million in FY 2024.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    report = validator.validate_metric(metric)
    assert report.is_valid is True
    assert len(report.errors) == 0


# ==============================================================================
# Financial Domain Rules Tests
# ==============================================================================


def test_revenue_ebitda_evidence_confusion() -> None:
    validator = FinancialValidator()
    # Metric labelled Revenue, but evidence clearly discusses EBITDA
    metric = ExtractedMetric(
        metric="Revenue",
        value=5000000.0,
        unit="EUR",
        year=2024,
        evidence="EBITDA amounted to EUR 5.0 million for the year.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    res = FinancialExtractionResult(document_name="doc.pdf", metrics=[metric])
    validated_res, summary = validator.validate_document_results(res)

    assert validated_res.metrics[0].verification_status == "flagged"
    assert any("exclusively discusses EBITDA" in msg for msg in validated_res.metrics[0].flag_reasons)


def test_ebitda_exceeds_revenue_relationship() -> None:
    validator = FinancialValidator()
    # EBITDA of 50M exceeds Revenue of 40M for the same year
    rev = ExtractedMetric(
        metric="Revenue",
        value=40000000.0,
        unit="EUR",
        year=2024,
        evidence="Total revenue was EUR 40 million.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    ebitda = ExtractedMetric(
        metric="EBITDA",
        value=50000000.0,
        unit="EUR",
        year=2024,
        evidence="EBITDA was EUR 50 million.",
        source=SourceReference(page=2, document="doc.pdf"),
    )
    res = FinancialExtractionResult(
        document_name="doc.pdf",
        financial_year=2024,
        currency="EUR",
        metrics=[rev, ebitda],
    )
    validated_res, summary = validator.validate_document_results(res)

    ebitda_m = next(m for m in validated_res.metrics if m.metric == "EBITDA")
    assert ebitda_m.verification_status == "flagged"
    assert any("exceeds reported Revenue" in r for r in ebitda_m.flag_reasons)


def test_invalid_and_divergent_financial_years() -> None:
    validator = FinancialValidator()
    metric_future = ExtractedMetric(
        metric="Revenue",
        value=1000000.0,
        unit="EUR",
        year=2065,
        evidence="Revenue EUR 1M.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    metric_old = ExtractedMetric(
        metric="EBITDA",
        value=200000.0,
        unit="EUR",
        year=2010,  # diverges from document primary year 2024
        evidence="EBITDA EUR 200k.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    res = FinancialExtractionResult(
        document_name="doc.pdf",
        financial_year=2024,
        metrics=[metric_future, metric_old],
    )
    validated_res, summary = validator.validate_document_results(res)

    assert validated_res.metrics[0].verification_status == "flagged"
    assert any("standard operating range" in r for r in validated_res.metrics[0].flag_reasons)

    assert validated_res.metrics[1].verification_status == "flagged"
    assert any("diverges significantly" in r for r in validated_res.metrics[1].flag_reasons)


def test_currency_mismatch_and_unrecognized() -> None:
    validator = FinancialValidator()
    m_usd = ExtractedMetric(
        metric="Revenue",
        value=1000000.0,
        unit="USD",  # Document is EUR
        year=2024,
        evidence="Revenue was $1M.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    m_unknown = ExtractedMetric(
        metric="EBIT",
        value=500000.0,
        unit="XYZ_COIN",  # Not ISO currency
        year=2024,
        evidence="EBIT was 500k in XYZ_COIN.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    res = FinancialExtractionResult(
        document_name="doc.pdf",
        currency="EUR",
        metrics=[m_usd, m_unknown],
    )
    validated_res, summary = validator.validate_document_results(res)

    assert any("differs from document currency" in r for r in validated_res.metrics[0].flag_reasons)
    assert any("not in standard ISO currency" in r for r in validated_res.metrics[1].flag_reasons)


def test_percentage_versus_monetary_confusion() -> None:
    validator = FinancialValidator()
    # Absolute metric with % unit
    m_bad_rev = ExtractedMetric(
        metric="Revenue",
        value=15.0,
        unit="%",
        year=2024,
        evidence="Revenue grew by 15%.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    # Growth metric with EUR unit
    m_bad_growth = ExtractedMetric(
        metric="Growth Percentage",
        value=12.5,
        unit="EUR",
        year=2024,
        evidence="Revenue growth reached 12.5%.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    res = FinancialExtractionResult(document_name="doc.pdf", metrics=[m_bad_rev, m_bad_growth])
    validated_res, summary = validator.validate_document_results(res)

    assert any("cannot have percentage unit '%'" in r for r in validated_res.metrics[0].flag_reasons)
    assert any("represents a rate/percentage but has unit" in r for r in validated_res.metrics[1].flag_reasons)


def test_null_value_preservation_and_review() -> None:
    validator = FinancialValidator()
    metric = ExtractedMetric(
        metric="Total Liabilities",
        value=None,  # missing value
        unit="EUR",
        year=2024,
        evidence="Total liabilities were not disclosed in the preliminary release.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    res = FinancialExtractionResult(document_name="doc.pdf", metrics=[metric])
    validated_res, summary = validator.validate_document_results(res)

    assert validated_res.metrics[0].value is None
    assert validated_res.metrics[0].verification_status == "flagged"
    assert any("requires manual confirmation" in r for r in validated_res.metrics[0].flag_reasons)


def test_suspicious_unit_scaling_detection() -> None:
    validator = FinancialValidator()
    # Evidence says "EUR 12.5 million", but value was not scaled (12.5)
    metric = ExtractedMetric(
        metric="Revenue",
        value=12.5,
        unit="EUR",
        year=2024,
        evidence="Total revenue rose to EUR 12.5 million.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    res = FinancialExtractionResult(document_name="doc.pdf", metrics=[metric])
    validated_res, summary = validator.validate_document_results(res)

    assert any("mentions 'million', but extracted value" in r for r in validated_res.metrics[0].flag_reasons)


def test_contradictory_metric_values_across_pages() -> None:
    validator = FinancialValidator()
    # Page 1 reports Revenue = 100M, Page 4 reports Revenue = 120M for the same year
    m1 = ExtractedMetric(
        metric="Revenue",
        value=100000000.0,
        unit="EUR",
        year=2024,
        evidence="Consolidated revenue reached EUR 100 million.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    m2 = ExtractedMetric(
        metric="Revenue",
        value=120000000.0,
        unit="EUR",
        year=2024,
        evidence="Adjusted revenue reached EUR 120 million.",
        source=SourceReference(page=4, document="doc.pdf"),
    )
    res = FinancialExtractionResult(document_name="doc.pdf", metrics=[m1, m2])
    validated_res, summary = validator.validate_document_results(res)

    assert len(summary.contradictions) == 1
    assert validated_res.metrics[0].verification_status == "flagged"
    assert validated_res.metrics[1].verification_status == "flagged"
    assert any("Contradictory values detected" in r for r in validated_res.metrics[0].flag_reasons)


def test_duplicate_metric_detection() -> None:
    validator = FinancialValidator()
    # Identical value repeated across pages
    m1 = ExtractedMetric(
        metric="Net Income",
        value=15000000.0,
        unit="EUR",
        year=2024,
        evidence="Net income was EUR 15 million.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    m2 = ExtractedMetric(
        metric="Net Income",
        value=15000000.0,
        unit="EUR",
        year=2024,
        evidence="Page 5 table confirms Net income was EUR 15 million.",
        source=SourceReference(page=5, document="doc.pdf"),
    )
    res = FinancialExtractionResult(document_name="doc.pdf", metrics=[m1, m2])
    validated_res, summary = validator.validate_document_results(res)

    assert len(summary.duplicates) == 1
    assert summary.duplicates[0]["pages"] == [1, 5]


def test_source_evidence_traceability_verification() -> None:
    validator = FinancialValidator()
    doc = ProcessedDocument(
        file_path="doc.pdf",
        file_name="doc.pdf",
        file_type="pdf",
        file_size_bytes=100,
        page_count=2,
        pages=[
            PageContent(
                page_number=1,
                text="In fiscal year 2024, Acme Corp achieved revenue of EUR 50 million.",
            ),
            PageContent(
                page_number=2,
                text="Operating expenses amounted to EUR 35 million.",
            ),
        ],
    )

    # Valid metric with real evidence on page 1
    m_valid = ExtractedMetric(
        metric="Revenue",
        value=50000000.0,
        unit="EUR",
        year=2024,
        evidence="Acme Corp achieved revenue of EUR 50 million.",
        source=SourceReference(page=1, document="doc.pdf"),
    )
    # Fabricated / unverified evidence claim on page 2
    m_unverified = ExtractedMetric(
        metric="EBITDA",
        value=15000000.0,
        unit="EUR",
        year=2024,
        evidence="EBITDA was EUR 15 million according to secret statements.",
        source=SourceReference(page=2, document="doc.pdf"),
    )
    # Metric claiming to be on non-existent page 99
    m_non_existent_page = ExtractedMetric(
        metric="Total Assets",
        value=80000000.0,
        unit="EUR",
        year=2024,
        evidence="Total assets stood at EUR 80 million.",
        source=SourceReference(page=99, document="doc.pdf"),
    )

    res = FinancialExtractionResult(
        document_name="doc.pdf",
        financial_year=2024,
        currency="EUR",
        metrics=[m_valid, m_unverified, m_non_existent_page],
    )
    validated_res, summary = validator.validate_document_results(res, document=doc)

    assert validated_res.metrics[0].verification_status == "verified"

    assert validated_res.metrics[1].verification_status == "flagged"
    assert any("could not be verified on page 2" in r for r in validated_res.metrics[1].flag_reasons)

    assert validated_res.metrics[2].verification_status == "flagged"
    assert any("does not exist in the document" in r for r in validated_res.metrics[2].flag_reasons)

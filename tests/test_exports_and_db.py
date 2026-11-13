"""Unit tests for Phase 5 exports and database repository operations."""

from __future__ import annotations

import io
import json
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from core.document_loader import ProcessedDocument
from database.db import DatabaseManager
from database.repository import DocumentRepository
from exports.export_manager import ExportManager
from schemas.financial_schema import ExtractedMetric, FinancialExtractionResult, SourceReference
from validation.financial_validator import ValidationFlag, ValidationSummary


@pytest.fixture
def sample_result() -> FinancialExtractionResult:
    """Fixture providing a populated FinancialExtractionResult."""
    return FinancialExtractionResult(
        document_name="annual_report_2024.pdf",
        company_name="Vanguard Energy SE",
        country="Germany",
        industry="Renewable Energy",
        reporting_period="FY 2024",
        financial_year=2024,
        currency="EUR",
        model_used="qwen2.5:latest",
        metrics=[
            ExtractedMetric(
                metric="Revenue",
                value=125500000.0,
                unit="EUR",
                year=2024,
                reporting_period="FY 2024",
                evidence="Consolidated revenue reached EUR 125.5 million.",
                source=SourceReference(page=2, document="annual_report_2024.pdf"),
                verification_status="verified",
            ),
            ExtractedMetric(
                metric="EBITDA",
                value=24000000.0,
                unit="EUR",
                year=2024,
                reporting_period="FY 2024",
                evidence="EBITDA increased to EUR 24.0 million.",
                source=SourceReference(page=3, document="annual_report_2024.pdf"),
                verification_status="verified",
            ),
            ExtractedMetric(
                metric="Growth Percentage",
                value=12.5,
                unit="%",
                year=2024,
                evidence="Revenue growth was 12.5% year over year.",
                source=SourceReference(page=2, document="annual_report_2024.pdf"),
                verification_status="flagged",
                flag_reasons=["Requires audit review"],
            ),
        ],
    )


# ==============================================================================
# Export Tests
# ==============================================================================


def test_export_manager_json(sample_result: FinancialExtractionResult) -> None:
    json_str = ExportManager.to_json_string(sample_result)
    assert "Vanguard Energy SE" in json_str
    parsed = json.loads(json_str)
    assert parsed["company_name"] == "Vanguard Energy SE"
    assert len(parsed["metrics"]) == 3

    json_bytes = ExportManager.to_json_bytes(sample_result)
    assert isinstance(json_bytes, bytes)
    assert b"Vanguard Energy SE" in json_bytes


def test_export_manager_csv(sample_result: FinancialExtractionResult) -> None:
    csv_str = ExportManager.to_csv_string(sample_result)
    assert "Metric,Value,Unit,Year" in csv_str
    assert "Revenue,125500000.0,EUR,2024" in csv_str

    csv_bytes = ExportManager.to_csv_bytes(sample_result)
    assert isinstance(csv_bytes, bytes)
    assert b"Revenue,125500000.0" in csv_bytes


def test_export_manager_dataframe(sample_result: FinancialExtractionResult) -> None:
    df = ExportManager.to_dataframe(sample_result)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df["Metric"]) == ["Revenue", "EBITDA", "Growth Percentage"]
    assert df["Value"].iloc[0] == 125500000.0


def test_export_manager_excel_multi_sheet(sample_result: FinancialExtractionResult) -> None:
    summary = ValidationSummary(
        total_metrics=3,
        verified_count=2,
        flagged_count=1,
        pending_count=0,
        flags=[
            ValidationFlag(
                metric_index=2,
                rule_name="audit_review",
                severity="warning",
                message="Requires audit review",
                metric_name="Growth Percentage",
                page_number=2,
            )
        ],
    )
    excel_bytes = ExportManager.to_excel_bytes(sample_result, validation_summary=summary)
    assert isinstance(excel_bytes, bytes)

    # Verify with openpyxl that sheets exist and have data
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    assert "Overview" in wb.sheetnames
    assert "Financial Metrics" in wb.sheetnames
    assert "Validation Audit" in wb.sheetnames

    overview_sheet = wb["Overview"]
    assert overview_sheet["A2"].value == "Document Name"
    assert overview_sheet["B2"].value == "annual_report_2024.pdf"


# ==============================================================================
# Database Repository Tests
# ==============================================================================


def test_database_repository_lifecycle(tmp_path: Path, sample_result: FinancialExtractionResult) -> None:
    db_path = tmp_path / "test_intel.db"
    db = DatabaseManager(db_path=db_path)
    repo = DocumentRepository(db_manager=db)

    # 1. Initial KPIs are 0
    kpis = repo.get_kpis()
    assert kpis["total_documents"] == 0
    assert kpis["total_metrics"] == 0

    # 2. Save document pipeline output
    doc = ProcessedDocument(
        file_path="annual_report_2024.pdf",
        file_name="annual_report_2024.pdf",
        file_type="pdf",
        file_size_bytes=50000,
        page_count=10,
    )
    summary = ValidationSummary(total_metrics=3, verified_count=2, flagged_count=1, pending_count=0)

    doc_id = repo.save_extraction_pipeline_output(doc, sample_result, summary)
    assert doc_id == 1

    # 3. Check KPIs after saving
    kpis = repo.get_kpis()
    assert kpis["total_documents"] == 1
    assert kpis["total_metrics"] == 3
    assert kpis["verified_metrics"] == 2
    assert kpis["flagged_metrics"] == 1

    # 4. List documents
    docs = repo.list_documents()
    assert len(docs) == 1
    assert docs[0]["filename"] == "annual_report_2024.pdf"
    assert docs[0]["company_name"] == "Vanguard Energy SE"
    assert docs[0]["metric_count"] == 3

    # 5. Get document by ID
    loaded = repo.get_document(doc_id)
    assert loaded is not None
    assert loaded["company_name"] == "Vanguard Energy SE"
    assert loaded["extraction_data"]["currency"] == "EUR"

    # 6. Get metrics for document
    metrics = repo.get_metrics_for_document(doc_id)
    assert len(metrics) == 3
    assert metrics[0]["metric"] == "Revenue"
    assert metrics[0]["value"] == 125500000.0

    # 7. Delete document
    assert repo.delete_document(doc_id) is True
    assert repo.get_document(doc_id) is None
    assert len(repo.list_documents()) == 0
    assert len(repo.get_metrics_for_document(doc_id)) == 0


def test_database_search_and_audit_logging(tmp_path: Path, sample_result: FinancialExtractionResult) -> None:
    db_path = tmp_path / "search_audit.db"
    db = DatabaseManager(db_path=db_path)
    repo = DocumentRepository(db_manager=db)

    # 1. Test audit logging
    log_id = repo.log_pipeline_event(
        event_type="test_event",
        message="Running automated unit test",
        level="INFO",
        details={"test": True},
    )
    assert log_id > 0
    logs = repo.get_audit_logs()
    assert len(logs) == 1
    assert logs[0]["event_type"] == "test_event"
    assert logs[0]["details"]["test"] is True

    # 2. Save two documents
    doc1 = ProcessedDocument(
        file_path="northstar_2024.pdf",
        file_name="northstar_2024.pdf",
        file_type="pdf",
        file_size_bytes=1000,
        page_count=2,
    )
    res1 = FinancialExtractionResult(
        document_name="northstar_2024.pdf",
        company_name="Northstar GmbH",
        financial_year=2024,
        metrics=[
            ExtractedMetric(
                metric="Revenue",
                value=10800000.0,
                unit="EUR",
                year=2024,
                evidence="Revenue EUR 10.8M",
                source=SourceReference(page=1, document="northstar_2024.pdf"),
                verification_status="verified",
            )
        ],
    )
    id1 = repo.save_extraction_pipeline_output(doc1, res1)

    doc2 = ProcessedDocument(
        file_path="northstar_2025.pdf",
        file_name="northstar_2025.pdf",
        file_type="pdf",
        file_size_bytes=1200,
        page_count=2,
    )
    res2 = FinancialExtractionResult(
        document_name="northstar_2025.pdf",
        company_name="Northstar GmbH",
        financial_year=2025,
        metrics=[
            ExtractedMetric(
                metric="Revenue",
                value=12500000.0,
                unit="EUR",
                year=2025,
                evidence="Revenue EUR 12.5M",
                source=SourceReference(page=1, document="northstar_2025.pdf"),
                verification_status="verified",
            ),
            ExtractedMetric(
                metric="EBITDA",
                value=2100000.0,
                unit="EUR",
                year=2025,
                evidence="EBITDA EUR 2.1M",
                source=SourceReference(page=1, document="northstar_2025.pdf"),
                verification_status="flagged",
            ),
        ],
    )
    id2 = repo.save_extraction_pipeline_output(doc2, res2)

    # 3. Test search documents
    search_res = repo.search_documents(query="northstar")
    assert len(search_res) == 2

    search_year = repo.search_documents(year=2025)
    assert len(search_year) == 1
    assert search_year[0]["filename"] == "northstar_2025.pdf"

    verified_only = repo.search_documents(status_filter="verified_only")
    assert len(verified_only) == 1
    assert verified_only[0]["filename"] == "northstar_2024.pdf"

    has_flags = repo.search_documents(status_filter="has_flags")
    assert len(has_flags) == 1
    assert has_flags[0]["filename"] == "northstar_2025.pdf"

    # 4. Test cross-document historical metric queries
    distinct_metrics = repo.get_distinct_metric_names()
    assert "Revenue" in distinct_metrics
    assert "EBITDA" in distinct_metrics

    revenue_trend = repo.get_historical_metrics_by_name("Revenue")
    assert len(revenue_trend) == 2
    assert revenue_trend[0]["year"] == 2024
    assert revenue_trend[0]["value"] == 10800000.0
    assert revenue_trend[1]["year"] == 2025
    assert revenue_trend[1]["value"] == 12500000.0

    # 5. Test stats and vacuum
    stats = db.get_database_stats()
    assert stats["document_count"] == 2
    assert stats["metric_count"] == 3
    assert stats["audit_log_count"] >= 3
    db.vacuum()

    # 6. Test clear all records
    assert repo.clear_all_records() is True
    assert repo.get_kpis()["total_documents"] == 0

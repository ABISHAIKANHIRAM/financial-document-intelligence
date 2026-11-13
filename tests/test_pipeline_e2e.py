"""End-to-end integration tests for the full Financial Document Intelligence pipeline.

Tests the full lifecycle:
Document Upload / Ingestion -> Chunking -> Extraction -> Validation ->
SQLite Persistence -> Search/Retrieval -> Multi-Format Exports (JSON, CSV, Excel).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.document_loader import DocumentLoader
from database.db import DatabaseManager
from database.repository import DocumentRepository
from exports.export_manager import ExportManager
from llm.financial_extractor import FinancialExtractor
from llm.ollama_client import OllamaClient
from schemas.financial_schema import ExtractedMetric, FinancialExtractionResult, SourceReference
from validation.financial_validator import FinancialValidator

DATASET_DIR = Path(__file__).resolve().parent.parent / "financial_document_test_dataset"


@pytest.fixture
def temp_db(tmp_path: Path) -> DatabaseManager:
    """Fixture providing an isolated SQLite database."""
    db_file = tmp_path / "test_pipeline.db"
    return DatabaseManager(db_path=db_file)


def test_full_pipeline_lifecycle_e2e(temp_db: DatabaseManager) -> None:
    """Test full pipeline from document loading to database persistence and multi-sheet export."""
    repo = DocumentRepository(db_manager=temp_db)

    # 1. Document Loading & Inspection
    pdf_path = DATASET_DIR / "01_northstar_annual_report_2025.pdf"
    assert pdf_path.exists()

    loader = DocumentLoader()
    doc = loader.load(pdf_path)

    assert doc.file_type == "pdf"
    assert doc.page_count == 2
    assert len(doc.chunks) >= 2

    # 2. Local LLM Extraction (with deterministic mock for Ollama)
    mock_ollama = MagicMock(spec=OllamaClient)
    mock_ollama.model = "qwen2.5:latest"
    mock_ollama.ensure_ready.return_value = "qwen2.5:latest"

    chunk1_data = {
        "company_name": "Northstar Components GmbH",
        "financial_year": 2025,
        "currency": "EUR",
        "metrics": [
            {
                "metric": "Revenue",
                "value": 12500000.0,
                "unit": "EUR",
                "year": 2025,
                "evidence": "Revenue EUR 12.5 million",
                "source": {"page": 1, "document": "01_northstar_annual_report_2025.pdf"},
            },
            {
                "metric": "EBITDA",
                "value": 2100000.0,
                "unit": "EUR",
                "year": 2025,
                "evidence": "EBITDA EUR 2.1 million",
                "source": {"page": 1, "document": "01_northstar_annual_report_2025.pdf"},
            },
        ],
    }

    chunk2_data = {
        "company_name": "Northstar Components GmbH",
        "financial_year": 2025,
        "currency": "EUR",
        "metrics": [
            {
                "metric": "Growth Percentage",
                "value": 15.7,
                "unit": "%",
                "year": 2025,
                "evidence": "revenue growth of 15.7%",
                "source": {"page": 2, "document": "01_northstar_annual_report_2025.pdf"},
            }
        ],
    }

    mock_ollama.generate.side_effect = [json.dumps(chunk1_data), json.dumps(chunk2_data)]

    extractor = FinancialExtractor(ollama_client=mock_ollama)
    raw_result = extractor.extract_from_document(doc)

    assert raw_result.company_name == "Northstar Components GmbH"
    assert len(raw_result.metrics) == 3

    # 3. Deterministic Financial Validation
    validator = FinancialValidator()
    validated_result, summary = validator.validate_document_results(raw_result, document=doc)

    assert summary.total_metrics == 3
    assert summary.is_clean is True
    for metric in validated_result.metrics:
        assert metric.verification_status == "verified"

    # 4. SQLite Persistence
    doc_id = repo.save_extraction_pipeline_output(
        document=doc,
        extraction_result=validated_result,
        validation_summary=summary,
    )
    assert doc_id > 0

    repo.log_pipeline_event(
        event_type="end_to_end_test",
        message="Full pipeline integration test succeeded",
        document_id=doc_id,
        level="INFO",
    )

    # 5. Database Retrieval & Verification
    stored_doc = repo.get_document(doc_id)
    assert stored_doc is not None
    assert stored_doc["company_name"] == "Northstar Components GmbH"
    assert stored_doc["financial_year"] == 2025

    stored_metrics = repo.get_metrics_for_document(doc_id)
    assert len(stored_metrics) == 3
    metric_names = [m["metric"] for m in stored_metrics]
    assert "Revenue" in metric_names
    assert "EBITDA" in metric_names
    assert "Growth Percentage" in metric_names

    # Check KPIs
    kpis = repo.get_kpis()
    assert kpis["total_documents"] == 1
    assert kpis["total_metrics"] == 3
    assert kpis["verified_metrics"] == 3
    assert kpis["flagged_metrics"] == 0

    # 6. Multi-Format Exports
    json_bytes = ExportManager.to_json_bytes(validated_result)
    assert len(json_bytes) > 0
    parsed_json = json.loads(json_bytes.decode("utf-8"))
    assert parsed_json["company_name"] == "Northstar Components GmbH"

    csv_bytes = ExportManager.to_csv_bytes(validated_result)
    assert len(csv_bytes) > 0
    csv_text = csv_bytes.decode("utf-8")
    assert "Revenue" in csv_text
    assert "12500000" in csv_text

    excel_bytes = ExportManager.to_excel_bytes(validated_result, validation_summary=summary)
    assert len(excel_bytes) > 0
    # Excel files start with PK zip header
    assert excel_bytes.startswith(b"PK")

    # 7. Document Deletion & Cascade Verification
    deleted = repo.delete_document(doc_id)
    assert deleted is True
    assert repo.get_document(doc_id) is None
    assert len(repo.get_metrics_for_document(doc_id)) == 0


def test_corrupted_file_error_handling(tmp_path: Path) -> None:
    """Verify that corrupt or unparseable files report errors without crashing."""
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4\nInvalid binary junk here that cannot be parsed\x00\xff")

    loader = DocumentLoader()
    doc = loader.load(corrupt_pdf)

    # Should report warning or errors gracefully
    assert len(doc.warnings) > 0 or doc.page_count == 0


def test_empty_csv_and_text_files(tmp_path: Path) -> None:
    """Verify handling of empty or blank files."""
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")

    loader = DocumentLoader()
    doc = loader.load(empty_csv)

    assert doc.file_type == "csv"
    assert doc.page_count == 1
    assert len(doc.tables) == 0


def test_database_maintenance_vacuum(temp_db: DatabaseManager) -> None:
    """Verify database vacuuming and statistics calculation."""
    stats = temp_db.get_database_stats()
    assert "document_count" in stats
    assert "metric_count" in stats
    assert "audit_log_count" in stats
    assert "size_bytes" in stats

    temp_db.vacuum()

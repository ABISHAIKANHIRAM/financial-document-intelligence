"""Evaluation and verification test suite against financial_document_test_dataset and ground_truth.json."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.document_loader import DocumentLoader, ProcessedDocument
from core.pdf_inspector import PdfInspector
from llm.financial_extractor import FinancialExtractor
from llm.ollama_client import OllamaClient
from schemas.financial_schema import ExtractedMetric, FinancialExtractionResult, SourceReference
from validation.financial_validator import FinancialValidator

DATASET_DIR = Path(__file__).resolve().parent.parent / "financial_document_test_dataset"


@pytest.fixture
def ground_truth_data() -> dict:
    """Load ground truth expected values."""
    gt_path = DATASET_DIR / "ground_truth.json"
    assert gt_path.exists(), f"ground_truth.json not found at {gt_path}"
    return json.loads(gt_path.read_text(encoding="utf-8"))


# ==============================================================================
# Document Inspection on Test Dataset
# ==============================================================================


def test_northstar_annual_report_inspection() -> None:
    pdf_path = DATASET_DIR / "01_northstar_annual_report_2025.pdf"
    assert pdf_path.exists()

    inspector = PdfInspector()
    report = inspector.inspect(pdf_path)

    assert report.page_count == 2
    assert report.is_encrypted is False
    assert len(report.text_page_numbers) == 2
    assert len(report.scanned_page_numbers) == 0
    assert report.is_predominantly_scanned is False

    # Page 1 contains the highlights table and narrative
    p1 = report.pages[0]
    assert p1.has_text is True
    assert p1.char_count > 500


def test_northstar_press_release_inspection() -> None:
    pdf_path = DATASET_DIR / "02_northstar_press_release_2025.pdf"
    assert pdf_path.exists()

    inspector = PdfInspector()
    report = inspector.inspect(pdf_path)

    assert report.page_count == 1
    assert len(report.text_page_numbers) == 1
    assert len(report.scanned_page_numbers) == 0


def test_alpine_works_scanned_inspection() -> None:
    pdf_path = DATASET_DIR / "03_alpine_works_scanned_statement.pdf"
    assert pdf_path.exists()

    inspector = PdfInspector()
    report = inspector.inspect(pdf_path)

    assert report.page_count == 1
    # Document is image-only, so page 1 must be flagged as scanned
    assert 1 in report.scanned_page_numbers
    assert report.is_predominantly_scanned is True


def test_omv_excel_ingestion_and_prioritization() -> None:
    excel_path = DATASET_DIR / "entire-omv-ar25.xlsx"
    assert excel_path.exists()

    loader = DocumentLoader()
    doc = loader.load(excel_path)

    assert doc.file_type == "xlsx"
    assert doc.page_count == 300
    assert len(doc.tables) == 300
    # Chunks are capped at max_chunks but prioritized for financial statements
    assert len(doc.chunks) == loader.chunker.max_chunks
    # Check that financial sheets were prioritized in chunks
    chunk_texts = " ".join(c.text for c in doc.chunks)
    assert "income statement" in chunk_texts.lower() or "sales revenues" in chunk_texts.lower() or "cash flow" in chunk_texts.lower()


# ==============================================================================
# Ground Truth Comparison & Metric Precision Tests
# ==============================================================================


def test_northstar_annual_report_ground_truth_alignment(ground_truth_data: dict) -> None:
    """Verify that simulated LLM extraction against Northstar Annual Report matches ground truth."""
    expected_doc = next(
        d for d in ground_truth_data["documents"] if d["file"] == "01_northstar_annual_report_2025.pdf"
    )
    expected = expected_doc["expected_metrics"]

    loader = DocumentLoader()
    doc = loader.load(DATASET_DIR / "01_northstar_annual_report_2025.pdf")

    # Mock Ollama returning clean extracted values matching the document content
    client = MagicMock(spec=OllamaClient)
    client.model = "qwen2.5:latest"
    client.ensure_ready.return_value = "qwen2.5:latest"

    chunk1_response = {
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
            {
                "metric": "EBIT",
                "value": 1600000.0,
                "unit": "EUR",
                "year": 2025,
                "evidence": "EBIT EUR 1.6 million",
                "source": {"page": 1, "document": "01_northstar_annual_report_2025.pdf"},
            },
            {
                "metric": "Net Income",
                "value": 1050000.0,
                "unit": "EUR",
                "year": 2025,
                "evidence": "Net income EUR 1.05 million",
                "source": {"page": 1, "document": "01_northstar_annual_report_2025.pdf"},
            },
            {
                "metric": "Total Assets",
                "value": 18400000.0,
                "unit": "EUR",
                "year": 2025,
                "evidence": "Total assets EUR 18.4 million",
                "source": {"page": 1, "document": "01_northstar_annual_report_2025.pdf"},
            },
            {
                "metric": "Total Liabilities",
                "value": 7200000.0,
                "unit": "EUR",
                "year": 2025,
                "evidence": "Total liabilities EUR 7.2 million",
                "source": {"page": 1, "document": "01_northstar_annual_report_2025.pdf"},
            },
            {
                "metric": "Cash Flow",
                "value": 1350000.0,
                "unit": "EUR",
                "year": 2025,
                "evidence": "Operating cash flow EUR 1.35 million",
                "source": {"page": 1, "document": "01_northstar_annual_report_2025.pdf"},
            },
            {
                "metric": "Number of Employees",
                "value": 84.0,
                "unit": "count",
                "year": 2025,
                "reporting_period": "year-end",
                "evidence": "Employees (year-end) 84",
                "source": {"page": 1, "document": "01_northstar_annual_report_2025.pdf"},
            },
        ],
    }

    chunk2_response = {
        "company_name": "Northstar Components GmbH",
        "financial_year": 2025,
        "currency": "EUR",
        "metrics": [
            {
                "metric": "Growth Percentage",
                "value": 15.7,
                "unit": "%",
                "year": 2025,
                "evidence": "The company reported revenue growth of 15.7% compared with FY 2024.",
                "source": {"page": 2, "document": "01_northstar_annual_report_2025.pdf"},
            },
            {
                "metric": "Number of Employees",
                "value": 81.0,
                "unit": "count",
                "year": 2025,
                "reporting_period": "average",
                "evidence": "The average number of employees during 2025 was 81.",
                "source": {"page": 2, "document": "01_northstar_annual_report_2025.pdf"},
            },
        ],
    }

    client.generate.side_effect = [json.dumps(chunk1_response), json.dumps(chunk2_response)]

    extractor = FinancialExtractor(ollama_client=client)
    res = extractor.extract_from_document(doc)

    validator = FinancialValidator()
    validated_res, summary = validator.validate_document_results(res, document=doc)

    # 1. Company Information Checks
    assert validated_res.company_name == expected_doc["company"]
    assert validated_res.financial_year == expected_doc["financial_year"]
    assert validated_res.currency == expected_doc["currency"]

    # 2. Metric Precision vs Ground Truth
    metrics_by_name = {m.metric: m for m in validated_res.metrics}

    assert metrics_by_name["Revenue"].value == expected["revenue"]
    assert metrics_by_name["Revenue"].source.page == 1
    assert metrics_by_name["Revenue"].verification_status == "verified"

    assert metrics_by_name["EBITDA"].value == expected["ebitda"]
    assert metrics_by_name["EBIT"].value == expected["ebit"]
    assert metrics_by_name["Net Income"].value == expected["net_income"]
    assert metrics_by_name["Total Assets"].value == expected["total_assets"]
    assert metrics_by_name["Total Liabilities"].value == expected["total_liabilities"]
    assert metrics_by_name["Cash Flow"].value == expected["operating_cash_flow"]
    assert metrics_by_name["Growth Percentage"].value == expected["revenue_growth_percent"]
    assert metrics_by_name["Growth Percentage"].unit == "%"
    assert metrics_by_name["Growth Percentage"].source.page == 2

    # 3. Distinction of Average vs Year-End Employees
    emp_metrics = [m for m in validated_res.metrics if m.metric == "Number of Employees"]
    assert len(emp_metrics) == 2
    emp_values = {m.value for m in emp_metrics}
    assert 84.0 in emp_values  # year-end
    assert 81.0 in emp_values  # average


def test_northstar_press_release_preserves_preliminary_status(ground_truth_data: dict) -> None:
    """Verify that preliminary unaudited figures are preserved and not merged."""
    expected_doc = next(
        d for d in ground_truth_data["documents"] if d["file"] == "02_northstar_press_release_2025.pdf"
    )
    expected = expected_doc["expected_metrics"]

    loader = DocumentLoader()
    doc = loader.load(DATASET_DIR / "02_northstar_press_release_2025.pdf")

    client = MagicMock(spec=OllamaClient)
    client.model = "qwen2.5:latest"
    client.ensure_ready.return_value = "qwen2.5:latest"

    pr_payload = {
        "company_name": "Northstar Components GmbH",
        "financial_year": 2025,
        "currency": "EUR",
        "reporting_period": "FY 2025 (Preliminary)",
        "metrics": [
            {
                "metric": "Revenue",
                "value": 12300000.0,
                "unit": "EUR",
                "year": 2025,
                "reporting_period": "preliminary",
                "evidence": "Northstar Components GmbH today announced preliminary, unaudited revenue of EUR 12.3 million",
                "source": {"page": 1, "document": "02_northstar_press_release_2025.pdf"},
            },
            {
                "metric": "EBITDA",
                "value": 2000000.0,
                "unit": "EUR",
                "year": 2025,
                "reporting_period": "preliminary",
                "evidence": "Management also stated that preliminary EBITDA was approximately EUR 2.0 million.",
                "source": {"page": 1, "document": "02_northstar_press_release_2025.pdf"},
            },
            {
                "metric": "Number of Employees",
                "value": 84.0,
                "unit": "count",
                "year": 2025,
                "reporting_period": "year-end",
                "evidence": "The company employed 84 people at 31 December 2025.",
                "source": {"page": 1, "document": "02_northstar_press_release_2025.pdf"},
            },
        ],
    }

    client.generate.return_value = json.dumps(pr_payload)
    extractor = FinancialExtractor(ollama_client=client)
    res = extractor.extract_from_document(doc)

    assert res.document_name == "02_northstar_press_release_2025.pdf"
    assert len(res.metrics) == 3
    # Check that values match preliminary expected ground truth
    rev_metric = next(m for m in res.metrics if m.metric == "Revenue")
    assert rev_metric.value == expected["preliminary_revenue"]
    assert rev_metric.value != 12500000.0  # Must not match final 12.5M
    assert rev_metric.source.document == "02_northstar_press_release_2025.pdf"

    ebitda_metric = next(m for m in res.metrics if m.metric == "EBITDA")
    assert ebitda_metric.value == expected["preliminary_ebitda"]


def test_alpine_works_scanned_ground_truth_alignment(ground_truth_data: dict) -> None:
    """Verify that OCR text extracted from scanned PDF matches Alpine Works ground truth metrics."""
    expected_doc = next(
        d for d in ground_truth_data["documents"] if d["file"] == "03_alpine_works_scanned_statement.pdf"
    )
    expected = expected_doc["expected_metrics"]

    loader = DocumentLoader()
    doc = loader.load(DATASET_DIR / "03_alpine_works_scanned_statement.pdf")

    # Verify OCR extracted chunks
    assert doc.page_count == 1
    assert len(doc.chunks) >= 1
    ocr_text = doc.chunks[0].text

    # Verify key figures exist in the OCR text
    assert "8,750,000" in ocr_text or "8750000" in ocr_text
    assert "1,420,000" in ocr_text or "1420000" in ocr_text

    # Mock LLM extracting from this OCR chunk
    client = MagicMock(spec=OllamaClient)
    client.model = "qwen2.5:latest"
    client.ensure_ready.return_value = "qwen2.5:latest"

    alpine_payload = {
        "company_name": "Alpine Works AG",
        "financial_year": 2025,
        "currency": "EUR",
        "reporting_period": "FY 2025",
        "metrics": [
            {
                "metric": "Revenue",
                "value": 8750000.0,
                "unit": "EUR",
                "year": 2025,
                "evidence": "Revenue EUR 8,750,000",
                "source": {"page": 1, "document": "03_alpine_works_scanned_statement.pdf"},
            },
            {
                "metric": "EBITDA",
                "value": 1420000.0,
                "unit": "EUR",
                "year": 2025,
                "evidence": "EBITDA EUR 1,420,000",
                "source": {"page": 1, "document": "03_alpine_works_scanned_statement.pdf"},
            },
            {
                "metric": "EBIT",
                "value": 1030000.0,
                "unit": "EUR",
                "year": 2025,
                "evidence": "EBIT EUR 1,030,000",
                "source": {"page": 1, "document": "03_alpine_works_scanned_statement.pdf"},
            },
            {
                "metric": "Net Income",
                "value": 645000.0,
                "unit": "EUR",
                "year": 2025,
                "evidence": "Net income EUR645,000",
                "source": {"page": 1, "document": "03_alpine_works_scanned_statement.pdf"},
            },
            {
                "metric": "Total Assets",
                "value": 11200000.0,
                "unit": "EUR",
                "year": 2025,
                "evidence": "Total assets EUR11,200,000",
                "source": {"page": 1, "document": "03_alpine_works_scanned_statement.pdf"},
            },
            {
                "metric": "Total Liabilities",
                "value": 4850000.0,
                "unit": "EUR",
                "year": 2025,
                "evidence": "Totalliabilities EUR4,850,000",
                "source": {"page": 1, "document": "03_alpine_works_scanned_statement.pdf"},
            },
            {
                "metric": "Number of Employees",
                "value": 56.0,
                "unit": "count",
                "year": 2025,
                "evidence": "Employees at year-end 56",
                "source": {"page": 1, "document": "03_alpine_works_scanned_statement.pdf"},
            },
        ],
    }

    client.generate.return_value = json.dumps(alpine_payload)
    extractor = FinancialExtractor(ollama_client=client)
    res = extractor.extract_from_document(doc)

    validator = FinancialValidator()
    validated_res, summary = validator.validate_document_results(res, document=doc)

    assert validated_res.company_name == "Alpine Works AG"
    assert validated_res.financial_year == 2025
    assert validated_res.currency == "EUR"

    metrics_by_name = {m.metric: m for m in validated_res.metrics}
    assert metrics_by_name["Revenue"].value == expected["revenue"]
    assert metrics_by_name["EBITDA"].value == expected["ebitda"]
    assert metrics_by_name["EBIT"].value == expected["ebit"]
    assert metrics_by_name["Net Income"].value == expected["net_income"]
    assert metrics_by_name["Total Assets"].value == expected["total_assets"]
    assert metrics_by_name["Total Liabilities"].value == expected["total_liabilities"]
    assert metrics_by_name["Number of Employees"].value == expected["employees_year_end"]


"""Comprehensive test suite for Phase 2 document inspection, extraction, and normalization."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import docx
import pandas as pd
from PIL import Image, ImageDraw
import pymupdf
import pytest

from config import Settings
from core.chunking import Chunker, DocumentChunk
from core.document_loader import DocumentLoader, ProcessedDocument
from core.document_normalizer import DocumentNormalizer
from core.ocr_processor import OcrProcessor, OcrResult
from core.pdf_inspector import PdfInspector
from core.table_extractor import TableExtractor, format_table_as_markdown
from core.text_extractor import TextExtractor


@pytest.fixture
def sample_selectable_pdf(tmp_path: Path) -> Path:
    """Create a synthetic 2-page PDF with selectable financial text."""
    pdf_path = tmp_path / "selectable_financial_report.pdf"
    doc = pymupdf.open()

    # Page 1: Narrative & Highlights
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        (50, 72),
        "Acme Global Holdings Inc.\n"
        "Annual Financial Report 2024\n\n"
        "Executive Summary:\n"
        "In fiscal year 2024, Acme Global Holdings achieved total revenue of EUR 125.5 million, "
        "representing an increase of 12.5% compared to fiscal year 2023.\n"
        "Operating profit (EBIT) reached EUR 18.2 million, and EBITDA was EUR 24.0 million.\n"
        "The company employed 1,450 full-time employees worldwide as of December 31, 2024.",
        fontsize=11,
    )

    # Page 2: Balance Sheet Highlights
    p2 = doc.new_page(width=595, height=842)
    p2.insert_text(
        (50, 72),
        "Financial Position Summary (Page 2)\n\n"
        "Total assets stood at EUR 85.0 million against total liabilities of EUR 32.5 million.\n"
        "Cash flow from operating activities was positive at EUR 15.3 million.\n"
        "Net income for the year was EUR 14.1 million.",
        fontsize=11,
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_scanned_pdf(tmp_path: Path) -> Path:
    """Create a synthetic PDF simulating a scanned document with an embedded image and no text layer."""
    pdf_path = tmp_path / "scanned_invoice.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)

    # Create a synthetic image of text using Pillow
    img = Image.new("RGB", (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "SCANNED VENDOR INVOICE", fill=(0, 0, 0))
    draw.text((20, 60), "Revenue: $500,000", fill=(0, 0, 0))

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # Insert image into PDF without selectable text
    rect = pymupdf.Rect(50, 50, 450, 250)
    page.insert_image(rect, stream=img_bytes.getvalue())

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_docx(tmp_path: Path) -> Path:
    """Create a synthetic Word document with paragraphs and a financial table."""
    docx_path = tmp_path / "quarterly_update.docx"
    doc = docx.Document()
    doc.add_heading("NovaTech Solutions - Q3 Press Release", level=1)
    doc.add_paragraph("NovaTech Solutions announced revenue of USD 45.8 million for Q3 2024.")
    doc.add_paragraph("Net income reached USD 6.2 million with EBITDA of USD 9.5 million.")

    table = doc.add_table(rows=3, cols=3)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Q3 2024"
    table.cell(0, 2).text = "Q3 2023"

    table.cell(1, 0).text = "Revenue"
    table.cell(1, 1).text = "45.8"
    table.cell(1, 2).text = "38.2"

    table.cell(2, 0).text = "EBITDA"
    table.cell(2, 1).text = "9.5"
    table.cell(2, 2).text = "7.8"

    doc.save(str(docx_path))
    return docx_path


@pytest.fixture
def sample_xlsx(tmp_path: Path) -> Path:
    """Create a synthetic Excel workbook with two financial sheets."""
    xlsx_path = tmp_path / "financial_model.xlsx"
    with pd.ExcelWriter(str(xlsx_path), engine="openpyxl") as writer:
        income_df = pd.DataFrame(
            {
                "Line Item": ["Revenue", "EBITDA", "EBIT", "Net Income"],
                "2024 (EUR)": [12000000, 2500000, 1800000, 1200000],
                "2023 (EUR)": [10500000, 2100000, 1500000, 950000],
            }
        )
        income_df.to_excel(writer, sheet_name="IncomeStatement", index=False)

        balance_df = pd.DataFrame(
            {
                "Item": ["Total Assets", "Total Liabilities"],
                "Amount (EUR)": [25000000, 11000000],
            }
        )
        balance_df.to_excel(writer, sheet_name="BalanceSheet", index=False)

    return xlsx_path


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a synthetic CSV file with financial metrics."""
    csv_path = tmp_path / "press_metrics.csv"
    content = (
        "Company,Year,Revenue,EBITDA,Currency\n"
        "Solaris AG,2024,88.4,14.2,EUR\n"
        "Solaris AG,2023,76.1,11.8,EUR\n"
    )
    csv_path.write_text(content, encoding="utf-8")
    return csv_path


# ==============================================================================
# PDF Inspection & Text Extraction Tests
# ==============================================================================


def test_pdf_inspector_detects_selectable_text(sample_selectable_pdf: Path) -> None:
    inspector = PdfInspector()
    report = inspector.inspect(sample_selectable_pdf)

    assert report.page_count == 2
    assert report.is_encrypted is False
    assert len(report.text_page_numbers) == 2
    assert len(report.scanned_page_numbers) == 0
    assert report.is_predominantly_scanned is False

    p1 = report.pages[0]
    assert p1.has_text is True
    assert p1.is_scanned is False
    assert p1.requires_ocr is False
    assert p1.char_count > 100


def test_pdf_inspector_detects_scanned_page(sample_scanned_pdf: Path) -> None:
    inspector = PdfInspector()
    report = inspector.inspect(sample_scanned_pdf)

    assert report.page_count == 1
    assert 1 in report.scanned_page_numbers
    assert report.is_predominantly_scanned is True

    p1 = report.pages[0]
    assert p1.is_scanned is True
    assert p1.image_count == 1
    assert p1.has_text is False


def test_text_extractor_extracts_page_content(sample_selectable_pdf: Path) -> None:
    extractor = TextExtractor()
    pages = extractor.extract_pdf_document(sample_selectable_pdf)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Acme Global Holdings Inc." in pages[0].text
    assert "EUR 125.5 million" in pages[0].text
    assert pages[1].page_number == 2
    assert "EUR 85.0 million" in pages[1].text


# ==============================================================================
# DOCX, XLSX, and CSV Extraction Tests
# ==============================================================================


def test_docx_extraction(sample_docx: Path) -> None:
    loader = DocumentLoader()
    doc = loader.load(sample_docx)

    assert doc.file_type == "docx"
    assert doc.page_count == 1
    assert len(doc.tables) == 1
    assert "NovaTech Solutions" in doc.get_full_text()
    assert "45.8" in doc.tables[0].markdown


def test_xlsx_extraction(sample_xlsx: Path) -> None:
    loader = DocumentLoader()
    doc = loader.load(sample_xlsx)

    assert doc.file_type == "xlsx"
    assert doc.page_count == 2
    assert len(doc.tables) == 2
    assert doc.tables[0].sheet_name == "IncomeStatement"
    assert "Revenue" in doc.tables[0].markdown
    assert "Total Assets" in doc.tables[1].markdown


def test_csv_extraction(sample_csv: Path) -> None:
    loader = DocumentLoader()
    doc = loader.load(sample_csv)

    assert doc.file_type == "csv"
    assert doc.page_count == 1
    assert len(doc.tables) == 1
    assert "Solaris AG" in doc.tables[0].markdown
    assert "88.4" in doc.tables[0].markdown


# ==============================================================================
# Normalization & Financial Formatting Tests
# ==============================================================================


def test_document_normalizer_cleans_unicode() -> None:
    normalizer = DocumentNormalizer()
    raw = "“Revenue” was EUR 10.5m – up 5% with \u00a0ligature \ufb01nance and minus −12.4%."
    cleaned = normalizer.normalize_text(raw)

    assert '"Revenue"' in cleaned
    assert " - " in cleaned
    assert "finance" in cleaned
    assert "-12.4%" in cleaned
    assert "\u00a0" not in cleaned


def test_normalize_financial_numbers() -> None:
    fn = DocumentNormalizer.normalize_financial_number

    assert fn("12,500,000") == 12500000.0
    assert fn("(1,250.50)") == -1250.50
    assert fn("-450.25") == -450.25
    assert fn("EUR 12.5M") == 12500000.0
    assert fn("$ 1.2B") == 1200000000.0
    assert fn("15.5%") == 15.5
    assert fn("invalid") is None


# ==============================================================================
# Chunking & Provenance Tests
# ==============================================================================


def test_chunker_preserves_page_numbers() -> None:
    chunker = Chunker(chunk_size=150, chunk_overlap=20)
    pages = [
        {"page_number": 1, "text": "Page one text that is relatively short.", "source_type": "text_layer"},
        {
            "page_number": 2,
            "text": "Page two text with paragraph 1.\n\nPage two paragraph 2 is longer and contains more details to trigger chunk splitting safely.",
            "source_type": "text_layer",
        },
    ]
    chunks = chunker.chunk_document(pages, document_name="test.pdf")

    assert len(chunks) >= 2
    assert chunks[0].page_number == 1
    assert chunks[0].document_name == "test.pdf"
    assert chunks[0].chunk_id.startswith("test.pdf_p1")
    assert any(c.page_number == 2 for c in chunks)


# ==============================================================================
# OCR Fallback & Error Handling Tests
# ==============================================================================


def test_ocr_processor_graceful_missing_binary() -> None:
    settings = Settings(ocr_enabled=True, tesseract_cmd="non_existent_binary_path_xyz")
    ocr = OcrProcessor(settings=settings)
    ocr._rapidocr_available = False  # Simulate absence of both OCR engines

    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    res = ocr.ocr_image(img, page_number=1)

    assert res.success is False
    assert "No OCR engine available" in (res.error_message or "")


def test_ocr_processor_mocked_success() -> None:
    ocr = OcrProcessor()
    ocr._tesseract_binary = "tesseract.exe"
    mock_data = {
        "text": ["Revenue", "EUR", "12.5m"],
        "conf": ["95.0", "92.0", "90.0"],
    }
    with patch("pytesseract.image_to_data", return_value=mock_data), \
         patch.object(ocr, "is_available", return_value=True):
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        res = ocr.ocr_image(img, page_number=3)

        assert res.success is True
        assert res.page_number == 3
        assert "Revenue EUR 12.5m" in res.text
        assert res.confidence > 0.90
        assert res.engine == "tesseract"


def test_ocr_processor_rapidocr_fallback() -> None:
    ocr = OcrProcessor()
    ocr._tesseract_binary = None  # Force RapidOCR path
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "Revenue EUR 5,000,000", fill=(0, 0, 0))

    res = ocr.ocr_image(img, page_number=1)
    assert res.success is True
    assert res.engine == "rapidocr"
    assert "Revenue" in res.text or "5,000,000" in res.text


def test_document_loader_records_unsupported_format(tmp_path: Path) -> None:
    bad_file = tmp_path / "archive.zip"
    bad_file.write_bytes(b"PK000fakezip")
    loader = DocumentLoader()

    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load(bad_file)


def test_document_loader_full_pipeline_pdf(sample_selectable_pdf: Path) -> None:
    loader = DocumentLoader()
    doc = loader.load(sample_selectable_pdf)

    assert isinstance(doc, ProcessedDocument)
    assert doc.page_count == 2
    assert doc.file_name == sample_selectable_pdf.name
    assert len(doc.chunks) > 0
    assert doc.chunks[0].page_number == 1
    assert "Acme Global Holdings Inc." in doc.get_full_text()
    assert doc.inspection_report is not None
    assert len(doc.errors) == 0

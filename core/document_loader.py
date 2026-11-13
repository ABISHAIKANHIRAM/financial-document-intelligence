"""Unified document loader for PDF, DOCX, XLSX, and CSV formats."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf

from config import Settings, get_settings
from core.chunking import Chunker, DocumentChunk
from core.document_normalizer import DocumentNormalizer
from core.ocr_processor import OcrProcessor
from core.pdf_inspector import PdfInspectionReport, PdfInspector
from core.table_extractor import ExtractedTable, TableExtractor
from core.text_extractor import TextExtractor

logger = logging.getLogger("financial_intel.document_loader")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".csv"}


@dataclass
class PageContent:
    """Consolidated content and provenance for a single document page or sheet."""

    page_number: int
    text: str  # Normalized text with integrated tables
    raw_text: str = ""
    tables: List[ExtractedTable] = field(default_factory=list)
    is_scanned: bool = False
    extraction_method: str = "text_layer"  # "text_layer", "ocr", "table", "docx", "xlsx", "csv"
    ocr_confidence: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert page content dataclass to dictionary."""
        return {
            "page_number": self.page_number,
            "text": self.text,
            "raw_text": self.raw_text,
            "tables": [t.to_dict() for t in self.tables],
            "is_scanned": self.is_scanned,
            "extraction_method": self.extraction_method,
            "ocr_confidence": self.ocr_confidence,
            "warnings": self.warnings,
        }


@dataclass
class ProcessedDocument:
    """Full extraction output of an ingested document ready for LLM extraction."""

    file_path: str
    file_name: str
    file_type: str
    file_size_bytes: int
    page_count: int
    pages: List[PageContent] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    chunks: List[DocumentChunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    inspection_report: Optional[PdfInspectionReport] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def get_page(self, page_number: int) -> Optional[PageContent]:
        """Lookup a page by 1-indexed page number."""
        for p in self.pages:
            if p.page_number == page_number:
                return p
        return None

    def get_full_text(self) -> str:
        """Return combined normalized text across all pages."""
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire processed document to dictionary."""
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "page_count": self.page_count,
            "pages": [p.to_dict() for p in self.pages],
            "tables": [t.to_dict() for t in self.tables],
            "chunks": [c.to_dict() for c in self.chunks],
            "metadata": self.metadata,
            "inspection_report": (
                self.inspection_report.to_dict() if self.inspection_report else None
            ),
            "warnings": self.warnings,
            "errors": self.errors,
        }


class DocumentLoader:
    """Loads, inspects, extracts, normalizes, and chunks financial documents."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        inspector: Optional[PdfInspector] = None,
        ocr_processor: Optional[OcrProcessor] = None,
        table_extractor: Optional[TableExtractor] = None,
        text_extractor: Optional[TextExtractor] = None,
        normalizer: Optional[DocumentNormalizer] = None,
        chunker: Optional[Chunker] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.inspector = inspector or PdfInspector(self.settings)
        self.ocr_processor = ocr_processor or OcrProcessor(self.settings)
        self.table_extractor = table_extractor or TableExtractor()
        self.text_extractor = text_extractor or TextExtractor()
        self.normalizer = normalizer or DocumentNormalizer()
        self.chunker = chunker or Chunker(self.settings)

    def load(self, file_path: str | Path) -> ProcessedDocument:
        """Ingest any supported financial document and return a ProcessedDocument.

        Supported extensions: .pdf, .docx, .xlsx, .xls, .csv.
        Never silently discards extraction errors.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Document file does not exist: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format '{suffix}'. Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
            )

        file_size = path.stat().st_size
        logger.info("Loading document '%s' (type=%s, size=%d bytes)", path.name, suffix, file_size)

        if suffix == ".pdf":
            return self._load_pdf(path, file_size)
        elif suffix == ".docx":
            return self._load_docx(path, file_size)
        elif suffix in {".xlsx", ".xls"}:
            return self._load_xlsx(path, file_size)
        elif suffix == ".csv":
            return self._load_csv(path, file_size)
        else:
            raise ValueError(f"Unhandled file extension: {suffix}")

    def _load_pdf(self, path: Path, file_size: int) -> ProcessedDocument:
        """Inspect and process a PDF document across text, tables, and OCR layers."""
        inspection_report = self.inspector.inspect(path)
        errors: List[str] = list(inspection_report.errors)
        warnings: List[str] = []

        # Extract all tables upfront
        pdf_tables = self.table_extractor.extract_pdf_tables(path)
        tables_by_page: Dict[int, List[ExtractedTable]] = {}
        for t in pdf_tables:
            tables_by_page.setdefault(t.page_number, []).append(t)

        pages: List[PageContent] = []

        try:
            doc = pymupdf.open(str(path))
            for idx in range(len(doc)):
                page_num = idx + 1
                page = doc[idx]
                page_warnings: List[str] = []
                page_tables = tables_by_page.get(page_num, [])

                # Determine if page was flagged as scanned during inspection
                page_inspection = next(
                    (p for p in inspection_report.pages if p.page_number == page_num),
                    None,
                )
                is_scanned = bool(page_inspection and page_inspection.is_scanned)

                raw_text = ""
                extraction_method = "text_layer"
                ocr_conf: Optional[float] = None

                if is_scanned and self.settings.ocr_enabled:
                    ocr_res = self.ocr_processor.ocr_pdf_page(page, page_number=page_num)
                    if ocr_res.success:
                        raw_text = ocr_res.text
                        extraction_method = "ocr"
                        ocr_conf = ocr_res.confidence
                    else:
                        page_warnings.append(
                            f"OCR failed on scanned page {page_num}: {ocr_res.error_message}"
                        )
                        # Fallback to direct text layer extraction even if sparse
                        raw_text = page.get_text("text") or ""
                        extraction_method = "sparse_text_fallback"
                else:
                    extracted_text = self.text_extractor.extract_pdf_page(page, page_number=page_num)
                    raw_text = extracted_text.text
                    page_warnings.extend(extracted_text.warnings)

                # Integrate narrative and tables
                from core.text_extractor import PageText
                temp_page_text = PageText(page_number=page_num, text=raw_text)
                integrated_text = self.normalizer.integrate_page_content(
                    temp_page_text, page_tables=page_tables
                )

                warnings.extend(page_warnings)

                pages.append(
                    PageContent(
                        page_number=page_num,
                        text=integrated_text,
                        raw_text=raw_text,
                        tables=page_tables,
                        is_scanned=is_scanned,
                        extraction_method=extraction_method,
                        ocr_confidence=ocr_conf,
                        warnings=page_warnings,
                    )
                )
            doc.close()
        except Exception as exc:
            err = f"Failed processing PDF pages for '{path.name}': {exc}"
            logger.error(err)
            errors.append(err)

        # Chunk the processed pages
        pages_payload = [
            {
                "page_number": p.page_number,
                "text": p.text,
                "source_type": p.extraction_method,
            }
            for p in pages
        ]
        chunks = self.chunker.chunk_document(pages_payload, document_name=path.name)

        return ProcessedDocument(
            file_path=str(path),
            file_name=path.name,
            file_type="pdf",
            file_size_bytes=file_size,
            page_count=len(pages),
            pages=pages,
            tables=pdf_tables,
            chunks=chunks,
            metadata=inspection_report.metadata,
            inspection_report=inspection_report,
            warnings=warnings,
            errors=errors,
        )

    def _load_docx(self, path: Path, file_size: int) -> ProcessedDocument:
        """Process a Microsoft Word document (.docx)."""
        warnings: List[str] = []
        errors: List[str] = []

        docx_text = self.text_extractor.extract_docx_document(path)
        docx_tables = self.table_extractor.extract_docx_tables(path)

        page_texts = docx_text if docx_text else []
        primary_text = page_texts[0] if page_texts else None
        raw_text = primary_text.text if primary_text else ""

        integrated_text = self.normalizer.integrate_page_content(
            primary_text, page_tables=docx_tables
        ) if primary_text else ""

        page = PageContent(
            page_number=1,
            text=integrated_text,
            raw_text=raw_text,
            tables=docx_tables,
            is_scanned=False,
            extraction_method="docx",
        )
        pages = [page]

        chunks = self.chunker.chunk_document(
            [{"page_number": 1, "text": integrated_text, "source_type": "docx"}],
            document_name=path.name,
        )

        return ProcessedDocument(
            file_path=str(path),
            file_name=path.name,
            file_type="docx",
            file_size_bytes=file_size,
            page_count=1,
            pages=pages,
            tables=docx_tables,
            chunks=chunks,
            metadata={"format": "docx"},
            warnings=warnings,
            errors=errors,
        )

    def _load_xlsx(self, path: Path, file_size: int) -> ProcessedDocument:
        """Process an Excel workbook (.xlsx, .xls) across all worksheets."""
        warnings: List[str] = []
        errors: List[str] = []

        tables = self.table_extractor.extract_xlsx_tables(path)
        pages: List[PageContent] = []

        for idx, table in enumerate(tables):
            page_num = idx + 1
            sheet_title = f"Sheet: {table.sheet_name or f'Sheet_{page_num}'}"
            integrated_text = f"[{sheet_title}]\n{table.markdown}\n[/{sheet_title}]"

            pages.append(
                PageContent(
                    page_number=page_num,
                    text=integrated_text,
                    raw_text=table.markdown,
                    tables=[table],
                    is_scanned=False,
                    extraction_method="xlsx",
                )
            )

        chunks = self.chunker.chunk_document(
            [
                {
                    "page_number": p.page_number,
                    "text": p.text,
                    "source_type": "xlsx",
                }
                for p in pages
            ],
            document_name=path.name,
        )

        return ProcessedDocument(
            file_path=str(path),
            file_name=path.name,
            file_type="xlsx",
            file_size_bytes=file_size,
            page_count=len(pages),
            pages=pages,
            tables=tables,
            chunks=chunks,
            metadata={"format": "xlsx", "sheet_count": len(tables)},
            warnings=warnings,
            errors=errors,
        )

    def _load_csv(self, path: Path, file_size: int) -> ProcessedDocument:
        """Process a delimited CSV file."""
        warnings: List[str] = []
        errors: List[str] = []

        tables = self.table_extractor.extract_csv_table(path)
        first_table = tables[0] if tables else None
        raw_text = first_table.markdown if first_table else ""
        integrated_text = f"[CSV File: {path.name}]\n{raw_text}\n[/CSV File]"

        page = PageContent(
            page_number=1,
            text=integrated_text,
            raw_text=raw_text,
            tables=tables,
            is_scanned=False,
            extraction_method="csv",
        )
        pages = [page]

        chunks = self.chunker.chunk_document(
            [{"page_number": 1, "text": integrated_text, "source_type": "csv"}],
            document_name=path.name,
        )

        return ProcessedDocument(
            file_path=str(path),
            file_name=path.name,
            file_type="csv",
            file_size_bytes=file_size,
            page_count=1,
            pages=pages,
            tables=tables,
            chunks=chunks,
            metadata={"format": "csv"},
            warnings=warnings,
            errors=errors,
        )

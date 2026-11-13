"""Core document processing, inspection, extraction, OCR, normalization, and chunking."""

from core.chunking import Chunker, DocumentChunk
from core.document_loader import DocumentLoader, PageContent, ProcessedDocument
from core.document_normalizer import DocumentNormalizer
from core.ocr_processor import OcrProcessor, OcrResult
from core.pdf_inspector import PageInspection, PdfInspectionReport, PdfInspector
from core.table_extractor import ExtractedTable, TableExtractor, format_table_as_markdown
from core.text_extractor import PageText, TextBlock, TextExtractor

__all__ = [
    "Chunker",
    "DocumentChunk",
    "DocumentLoader",
    "DocumentNormalizer",
    "ExtractedTable",
    "OcrProcessor",
    "OcrResult",
    "PageContent",
    "PageInspection",
    "PageText",
    "PdfInspectionReport",
    "PdfInspector",
    "ProcessedDocument",
    "TableExtractor",
    "TextBlock",
    "TextExtractor",
    "format_table_as_markdown",
]

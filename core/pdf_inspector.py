"""PDF inspection module for detecting text layers, scanned pages, and metadata."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf

from config import Settings, get_settings

logger = logging.getLogger("financial_intel.pdf_inspector")


@dataclass(frozen=True)
class PageInspection:
    """Detailed inspection metrics for a single PDF page."""

    page_number: int  # 1-indexed for human / financial reference
    has_text: bool
    char_count: int
    word_count: int
    image_count: int
    drawing_count: int
    is_scanned: bool
    requires_ocr: bool
    width: float
    height: float
    rotation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert page inspection dataclass to dictionary."""
        return asdict(self)


@dataclass
class PdfInspectionReport:
    """Comprehensive inspection report for a PDF document."""

    file_path: str
    file_name: str
    page_count: int
    is_encrypted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: List[PageInspection] = field(default_factory=list)
    scanned_page_numbers: List[int] = field(default_factory=list)
    text_page_numbers: List[int] = field(default_factory=list)
    is_predominantly_scanned: bool = False
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert inspection report dataclass to dictionary."""
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "page_count": self.page_count,
            "is_encrypted": self.is_encrypted,
            "metadata": self.metadata,
            "pages": [p.to_dict() for p in self.pages],
            "scanned_page_numbers": self.scanned_page_numbers,
            "text_page_numbers": self.text_page_numbers,
            "is_predominantly_scanned": self.is_predominantly_scanned,
            "errors": self.errors,
        }


class PdfInspector:
    """Inspects PDF files to identify selectable text vs scanned pages and extract metadata."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.min_text_chars = self.settings.min_text_chars_per_page

    def inspect(self, file_path: str | Path) -> PdfInspectionReport:
        """Inspect a PDF file and return a structured PdfInspectionReport.

        Args:
            file_path: Absolute or relative path to the PDF document.

        Returns:
            PdfInspectionReport containing document metadata and page-by-page analysis.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"PDF document not found: {path}")

        report = PdfInspectionReport(
            file_path=str(path),
            file_name=path.name,
            page_count=0,
        )

        try:
            doc = pymupdf.open(str(path))
        except Exception as exc:
            err = f"Failed to open PDF '{path.name}': {exc}"
            logger.error(err)
            report.errors.append(err)
            return report

        try:
            report.is_encrypted = doc.is_encrypted
            if doc.is_encrypted:
                report.errors.append(
                    f"PDF '{path.name}' is password-protected or encrypted."
                )
                return report

            report.page_count = len(doc)
            report.metadata = self._extract_metadata(doc)

            for idx in range(len(doc)):
                page_num = idx + 1
                try:
                    page = doc[idx]
                    inspection = self.inspect_page(page, page_number=page_num)
                    report.pages.append(inspection)
                    if inspection.is_scanned:
                        report.scanned_page_numbers.append(page_num)
                    else:
                        report.text_page_numbers.append(page_num)
                except Exception as page_exc:
                    err = f"Failed inspecting page {page_num} of '{path.name}': {page_exc}"
                    logger.warning(err)
                    report.errors.append(err)

            if report.page_count > 0:
                scanned_ratio = len(report.scanned_page_numbers) / report.page_count
                report.is_predominantly_scanned = scanned_ratio >= 0.5

            logger.info(
                "PDF inspection complete for %s: %d total pages (%d text, %d scanned)",
                path.name,
                report.page_count,
                len(report.text_page_numbers),
                len(report.scanned_page_numbers),
            )
            return report
        finally:
            doc.close()

    def inspect_page(self, page: pymupdf.Page, page_number: int) -> PageInspection:
        """Inspect a single PyMuPDF page for text layer, images, and drawings.

        Args:
            page: PyMuPDF Page instance.
            page_number: 1-indexed page number.

        Returns:
            PageInspection with classification of text vs scanned content.
        """
        text = page.get_text("text") or ""
        char_count = len(text.strip())
        words = text.split()
        word_count = len(words)

        images = page.get_images(full=True) or []
        image_count = len(images)

        drawings = page.get_drawings() or []
        drawing_count = len(drawings)

        rect = page.rect
        width = float(rect.width)
        height = float(rect.height)
        rotation = int(page.rotation)

        # Scanned page detection heuristic:
        # A page is marked as scanned if selectable text character count is below threshold
        # AND it contains at least one image or contains essentially zero text.
        has_text = char_count >= self.min_text_chars
        is_scanned = (char_count < self.min_text_chars) and (image_count > 0 or char_count == 0)
        requires_ocr = is_scanned and self.settings.ocr_enabled

        return PageInspection(
            page_number=page_number,
            has_text=has_text,
            char_count=char_count,
            word_count=word_count,
            image_count=image_count,
            drawing_count=drawing_count,
            is_scanned=is_scanned,
            requires_ocr=requires_ocr,
            width=width,
            height=height,
            rotation=rotation,
        )

    @staticmethod
    def _extract_metadata(doc: pymupdf.Document) -> Dict[str, Any]:
        """Extract sanitized document metadata dictionary."""
        raw = doc.metadata or {}
        return {
            "title": (raw.get("title") or "").strip(),
            "author": (raw.get("author") or "").strip(),
            "subject": (raw.get("subject") or "").strip(),
            "keywords": (raw.get("keywords") or "").strip(),
            "creator": (raw.get("creator") or "").strip(),
            "producer": (raw.get("producer") or "").strip(),
            "creation_date": (raw.get("creationDate") or "").strip(),
            "mod_date": (raw.get("modDate") or "").strip(),
            "format": (raw.get("format") or "").strip(),
        }

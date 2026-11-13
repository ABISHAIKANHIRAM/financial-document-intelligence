"""Text extraction module for PDFs (PyMuPDF / pdfplumber fallback) and DOCX."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber
import pymupdf

logger = logging.getLogger("financial_intel.text_extractor")


@dataclass
class TextBlock:
    """A bounded text segment within a document page."""

    text: str
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1]
    block_type: int = 0  # 0 for text, 1 for image in PyMuPDF


@dataclass
class PageText:
    """Extracted text content and metadata for a single document page."""

    page_number: int
    text: str
    char_count: int = 0
    word_count: int = 0
    blocks: List[TextBlock] = field(default_factory=list)
    source_method: str = "pymupdf"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert page text dataclass to dictionary."""
        return {
            "page_number": self.page_number,
            "text": self.text,
            "char_count": self.char_count,
            "word_count": self.word_count,
            "blocks": [asdict(b) for b in self.blocks],
            "source_method": self.source_method,
            "warnings": self.warnings,
        }


class TextExtractor:
    """Extracts raw text content while preserving page indices and source references."""

    def extract_pdf_page(
        self,
        page: pymupdf.Page,
        page_number: int,
        use_fallback_plumber: bool = True,
        plumber_page: Optional[pdfplumber.page.Page] = None,
    ) -> PageText:
        """Extract text and layout blocks from a PyMuPDF Page with optional pdfplumber fallback."""
        try:
            raw_blocks = page.get_text("blocks") or []
            blocks: List[TextBlock] = []
            block_texts: List[str] = []

            for b in raw_blocks:
                # PyMuPDF block tuple: (x0, y0, x1, y1, text, block_no, block_type)
                if len(b) >= 5:
                    b_text = str(b[4]).strip()
                    bbox = [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
                    b_type = int(b[6]) if len(b) >= 7 else 0
                    if b_text:
                        blocks.append(TextBlock(text=b_text, bbox=bbox, block_type=b_type))
                        block_texts.append(b_text)

            combined_text = "\n\n".join(block_texts) if block_texts else (page.get_text("text") or "")
            clean_text = combined_text.strip()

            # If PyMuPDF extracted minimal text but pdfplumber is available, try fallback
            if len(clean_text) < 20 and use_fallback_plumber and plumber_page is not None:
                try:
                    fallback_text = plumber_page.extract_text() or ""
                    if len(fallback_text.strip()) > len(clean_text):
                        clean_text = fallback_text.strip()
                        return PageText(
                            page_number=page_number,
                            text=clean_text,
                            char_count=len(clean_text),
                            word_count=len(clean_text.split()),
                            blocks=blocks,
                            source_method="pdfplumber_fallback",
                        )
                except Exception as fb_exc:
                    logger.debug("pdfplumber fallback failed on page %d: %s", page_number, fb_exc)

            return PageText(
                page_number=page_number,
                text=clean_text,
                char_count=len(clean_text),
                word_count=len(clean_text.split()),
                blocks=blocks,
                source_method="pymupdf",
            )
        except Exception as exc:
            err = f"Failed extracting text from page {page_number}: {exc}"
            logger.error(err)
            return PageText(
                page_number=page_number,
                text="",
                source_method="failed",
                warnings=[err],
            )

    def extract_pdf_document(self, file_path: str | Path) -> List[PageText]:
        """Extract text from all pages of a PDF document."""
        path = Path(file_path).resolve()
        pages: List[PageText] = []

        try:
            doc = pymupdf.open(str(path))
            # Open pdfplumber optionally for fallback
            plumber_doc = None
            try:
                plumber_doc = pdfplumber.open(str(path))
            except Exception:
                pass

            try:
                for idx in range(len(doc)):
                    page_num = idx + 1
                    page = doc[idx]
                    plumber_page = (
                        plumber_doc.pages[idx]
                        if (plumber_doc and idx < len(plumber_doc.pages))
                        else None
                    )
                    extracted = self.extract_pdf_page(
                        page,
                        page_number=page_num,
                        plumber_page=plumber_page,
                    )
                    pages.append(extracted)
            finally:
                doc.close()
                if plumber_doc:
                    plumber_doc.close()
        except Exception as exc:
            logger.error("Failed opening PDF '%s' for text extraction: %s", path.name, exc)
            pages.append(
                PageText(
                    page_number=1,
                    text="",
                    source_method="failed",
                    warnings=[f"Failed opening PDF '{path.name}': {exc}"],
                )
            )

        return pages

    def extract_docx_document(self, file_path: str | Path) -> List[PageText]:
        """Extract paragraphs and headings from a Word (.docx) document."""
        import docx

        path = Path(file_path).resolve()
        try:
            doc = docx.Document(str(path))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            combined = "\n\n".join(paragraphs)

            return [
                PageText(
                    page_number=1,
                    text=combined,
                    char_count=len(combined),
                    word_count=len(combined.split()),
                    source_method="python-docx",
                )
            ]
        except Exception as exc:
            logger.error("Failed reading DOCX '%s': %s", path.name, exc)
            return [
                PageText(
                    page_number=1,
                    text="",
                    source_method="failed",
                    warnings=[f"Failed reading DOCX '{path.name}': {exc}"],
                )
            ]

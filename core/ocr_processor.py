"""OCR processing module supporting Tesseract and in-process ONNX Runtime (RapidOCR)."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import pymupdf
import pytesseract

from config import Settings, get_settings

logger = logging.getLogger("financial_intel.ocr")


@dataclass
class OcrResult:
    """Result of an OCR execution on a rendered document page or image."""

    page_number: int
    text: str = ""
    success: bool = True
    confidence: float = 0.0
    engine: str = "none"  # "tesseract", "rapidocr", "none"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert OCR result to dictionary."""
        return {
            "page_number": self.page_number,
            "text": self.text,
            "success": self.success,
            "confidence": self.confidence,
            "engine": self.engine,
            "error_message": self.error_message,
        }


class OcrProcessor:
    """Handles PDF page rendering and optical character recognition fallback.

    Supports:
    1. System Tesseract OCR (via pytesseract)
    2. Local in-process ONNX OCR (via RapidOCR)
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.language = self.settings.ocr_language
        self.dpi = self.settings.ocr_dpi
        self._tesseract_binary = self._resolve_tesseract_binary()
        self._rapidocr_engine = None
        self._rapidocr_available = self._check_rapidocr()

    def _resolve_tesseract_binary(self) -> Optional[str]:
        """Locate the tesseract executable on PATH or via explicit configuration."""
        if self.settings.tesseract_cmd:
            custom_path = Path(self.settings.tesseract_cmd)
            if custom_path.is_file():
                pytesseract.pytesseract.tesseract_cmd = str(custom_path)
                return str(custom_path)

        # Check standard PATH
        found = shutil.which("tesseract")
        if found:
            pytesseract.pytesseract.tesseract_cmd = found
            return found

        # Check standard Windows installation directories
        common_windows_paths = [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            Path.home() / r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        ]
        for candidate in common_windows_paths:
            if candidate.is_file():
                pytesseract.pytesseract.tesseract_cmd = str(candidate)
                return str(candidate)

        return None

    def _check_rapidocr(self) -> bool:
        """Check if local RapidOCR ONNX package is importable."""
        try:
            import rapidocr_onnxruntime  # noqa: F401
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        """Return True if OCR is enabled and either Tesseract or RapidOCR is ready."""
        if not self.settings.ocr_enabled:
            return False
        if self._tesseract_binary:
            try:
                if pytesseract.get_tesseract_version() is not None:
                    return True
            except Exception:
                pass
        return self._rapidocr_available

    def render_pdf_page_to_image(
        self,
        page: pymupdf.Page,
        dpi: Optional[int] = None,
    ) -> Image.Image:
        """Render a PyMuPDF PDF page into a high-resolution PIL Image.

        Uses PyMuPDF's built-in vector rasterizer, eliminating any dependency on external Poppler binaries.
        """
        target_dpi = dpi or self.dpi
        zoom = target_dpi / 72.0
        matrix = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return img

    def _get_rapidocr(self):
        """Lazily initialize RapidOCR engine."""
        if self._rapidocr_engine is None and self._rapidocr_available:
            from rapidocr_onnxruntime import RapidOCR
            self._rapidocr_engine = RapidOCR()
        return self._rapidocr_engine

    def ocr_image(self, image: Image.Image, page_number: int = 1) -> OcrResult:
        """Extract text and confidence score from a PIL Image using Tesseract or RapidOCR fallback."""
        if not self.settings.ocr_enabled:
            return OcrResult(
                page_number=page_number,
                success=False,
                error_message="OCR is disabled in configuration (OCR_ENABLED=false).",
            )

        # Priority 1: System Tesseract (if available)
        if self._tesseract_binary:
            try:
                custom_config = "--oem 3 --psm 3"
                data = pytesseract.image_to_data(
                    image,
                    lang=self.language,
                    config=custom_config,
                    output_type=pytesseract.Output.DICT,
                )

                texts: List[str] = []
                confidences: List[float] = []

                for i in range(len(data["text"])):
                    w_text = data["text"][i].strip()
                    conf_str = data["conf"][i]
                    try:
                        conf_val = float(conf_str)
                    except (ValueError, TypeError):
                        conf_val = -1.0

                    if w_text:
                        texts.append(w_text)
                        if conf_val >= 0:
                            confidences.append(conf_val)

                raw_text = " ".join(texts)
                avg_conf = (
                    round(sum(confidences) / len(confidences) / 100.0, 3)
                    if confidences
                    else 0.0
                )

                logger.info(
                    "Tesseract OCR processed page %d: %d characters, confidence=%.2f",
                    page_number,
                    len(raw_text),
                    avg_conf,
                )
                return OcrResult(
                    page_number=page_number,
                    text=raw_text,
                    success=True,
                    confidence=avg_conf,
                    engine="tesseract",
                )
            except Exception as tess_exc:
                logger.warning("Tesseract failed on page %d: %s; trying RapidOCR.", page_number, tess_exc)

        # Priority 2: In-Process RapidOCR (ONNX Runtime)
        if self._rapidocr_available:
            try:
                engine = self._get_rapidocr()
                img_np = np.array(image)
                results, _ = engine(img_np)

                if results:
                    lines = [r[1] for r in results if len(r) >= 2 and r[1].strip()]
                    confs = [float(r[2]) for r in results if len(r) >= 3 and r[2] is not None]
                    raw_text = "\n".join(lines)
                    avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.95

                    logger.info(
                        "RapidOCR processed page %d: %d characters, confidence=%.2f",
                        page_number,
                        len(raw_text),
                        avg_conf,
                    )
                    return OcrResult(
                        page_number=page_number,
                        text=raw_text,
                        success=True,
                        confidence=avg_conf,
                        engine="rapidocr",
                    )
                else:
                    return OcrResult(
                        page_number=page_number,
                        text="",
                        success=True,
                        confidence=1.0,
                        engine="rapidocr",
                    )
            except Exception as rapid_exc:
                logger.error("RapidOCR failed on page %d: %s", page_number, rapid_exc)

        # Neither OCR engine succeeded or available
        err = (
            "No OCR engine available. Install Tesseract OCR on your system "
            "or ensure rapidocr-onnxruntime is installed in Python environment."
        )
        logger.warning(err)
        return OcrResult(
            page_number=page_number,
            success=False,
            error_message=err,
        )

    def ocr_pdf_page(
        self,
        page: pymupdf.Page,
        page_number: int,
        dpi: Optional[int] = None,
    ) -> OcrResult:
        """Render a PyMuPDF page and perform OCR."""
        try:
            image = self.render_pdf_page_to_image(page, dpi=dpi)
            return self.ocr_image(image, page_number=page_number)
        except Exception as exc:
            err = f"Failed rendering page {page_number} for OCR: {exc}"
            logger.error(err)
            return OcrResult(
                page_number=page_number,
                success=False,
                error_message=err,
            )

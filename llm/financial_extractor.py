"""Local LLM financial extraction orchestrator with JSON repair and provenance preservation."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from config import Settings, get_settings
from core.chunking import DocumentChunk
from core.document_loader import ProcessedDocument
from llm.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
)
from llm.prompts import (
    FINANCIAL_EXTRACTION_SYSTEM_PROMPT,
    build_chunk_extraction_prompt,
    build_corrective_repair_prompt,
)
from schemas.financial_schema import (
    ChunkExtractionResponse,
    ExtractedMetric,
    FinancialExtractionResult,
    SourceReference,
)

logger = logging.getLogger("financial_intel.financial_extractor")

FINANCIAL_KEYWORDS: Set[str] = {
    "revenue",
    "turnover",
    "ebitda",
    "ebit",
    "operating profit",
    "net income",
    "profit",
    "loss",
    "assets",
    "liabilities",
    "cash flow",
    "employees",
    "headcount",
    "growth",
    "margin",
    "fiscal",
    "balance sheet",
    "income statement",
    "financial",
    "sales",
    "eur",
    "usd",
    "€",
    "$",
    "£",
}


def repair_json_string(raw_text: str) -> Optional[Dict[str, Any]]:
    """Attempt controlled multi-pass repair of a malformed LLM JSON string.

    Handles:
    - Markdown code fences (```json ... ```)
    - Surrounding conversational text
    - Trailing commas in objects and arrays
    - Common invalid escape sequences
    """
    if not raw_text or not raw_text.strip():
        return None

    cleaned = raw_text.strip()

    # Pass 1: Strip markdown code blocks
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Pass 2: Direct json parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Pass 3: Extract innermost or outermost JSON object {...}
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            # Pass 4: Fix trailing commas
            fixed = re.sub(r",\s*([\}\]])", r"\1", candidate)
            try:
                data = json.loads(fixed)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

    return None


class FinancialExtractor:
    """Orchestrates structured financial information extraction using a local Ollama model."""

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = ollama_client or OllamaClient()

    def is_chunk_relevant(self, chunk: DocumentChunk) -> bool:
        """Heuristic check to prioritize chunks containing financial terms or tables."""
        text_lower = chunk.text.lower()
        if "[table" in text_lower or "[/table" in text_lower:
            return True
        return any(kw in text_lower for kw in FINANCIAL_KEYWORDS)

    def extract_from_chunk(
        self,
        chunk: DocumentChunk,
        model: Optional[str] = None,
        retry_on_invalid_json: bool = True,
    ) -> Tuple[Optional[ChunkExtractionResponse], Optional[str]]:
        """Extract structured data from a single document chunk via local Ollama.

        Returns:
            Tuple of (ChunkExtractionResponse or None, error_message or None)
        """
        prompt = build_chunk_extraction_prompt(
            chunk_text=chunk.text,
            document_name=chunk.document_name,
            page_number=chunk.page_number,
            source_type=chunk.source_types[0] if chunk.source_types else "text_layer",
        )

        try:
            raw_response = self.client.generate(
                prompt=prompt,
                system=FINANCIAL_EXTRACTION_SYSTEM_PROMPT,
                model=model,
                response_format="json",
            )
        except Exception as exc:
            err = f"Ollama generation failed for chunk {chunk.chunk_id}: {exc}"
            logger.error(err)
            return None, err

        # Attempt to parse and repair JSON response
        parsed = repair_json_string(raw_response)
        error_detail = None

        if parsed is None and retry_on_invalid_json:
            logger.warning(
                "Invalid JSON from Ollama on chunk %s; attempting corrective retry.",
                chunk.chunk_id,
            )
            repair_prompt = build_corrective_repair_prompt(
                raw_response=raw_response,
                error_message="Output was not valid JSON",
            )
            try:
                retry_response = self.client.generate(
                    prompt=repair_prompt,
                    system=FINANCIAL_EXTRACTION_SYSTEM_PROMPT,
                    model=model,
                    response_format="json",
                )
                parsed = repair_json_string(retry_response)
            except Exception as retry_exc:
                error_detail = f"Retry prompt failed: {retry_exc}"

        if parsed is None:
            err = (
                f"LLM returned unparseable JSON for chunk {chunk.chunk_id}. "
                f"Raw snippet: {raw_response[:200]!r}"
            )
            logger.warning(err)
            return None, err

        # Validate with Pydantic ChunkExtractionResponse
        try:
            validated = ChunkExtractionResponse.model_validate(parsed)

            # Strictly enforce source traceability on every extracted metric
            for m in validated.metrics:
                # Ensure the page number matches the authentic source chunk page
                m.source = SourceReference(
                    page=chunk.page_number,
                    document=chunk.document_name,
                    source_type=chunk.source_types[0] if chunk.source_types else "text_layer",
                )
                m.extraction_method = "llm_extraction"

            return validated, None
        except Exception as val_exc:
            err = f"Schema validation error on chunk {chunk.chunk_id}: {val_exc}"
            logger.warning(err)
            return None, err

    def extract_from_document(
        self,
        document: ProcessedDocument,
        model: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> FinancialExtractionResult:
        """Extract all financial information and company metadata across a document's chunks.

        Args:
            document: ProcessedDocument instance from Phase 2 loader.
            model: Optional local Ollama model override.
            progress_callback: Optional callable(chunk_idx, total_chunks, status_msg).

        Returns:
            FinancialExtractionResult with aggregated company details and traceable metrics.
        """
        active_model = model or self.client.model
        # Check Ollama readiness before running extraction
        try:
            resolved_model = self.client.ensure_ready(active_model)
        except (OllamaConnectionError, OllamaModelNotFoundError) as exc:
            logger.error("Local Ollama is not ready: %s", exc)
            return FinancialExtractionResult(
                document_name=document.file_name,
                model_used=active_model,
                extraction_errors=[str(exc)],
            )

        # Prioritize relevant chunks
        all_chunks = document.chunks
        relevant_chunks = [c for c in all_chunks if self.is_chunk_relevant(c)]
        chunks_to_process = relevant_chunks if relevant_chunks else all_chunks

        total_to_process = len(chunks_to_process)
        logger.info(
            "Extracting financial info from '%s' using model '%s': %d of %d chunks selected",
            document.file_name,
            resolved_model,
            total_to_process,
            len(all_chunks),
        )

        all_metrics: List[ExtractedMetric] = []
        company_names: List[str] = []
        countries: List[str] = []
        industries: List[str] = []
        reporting_periods: List[str] = []
        financial_years: List[int] = []
        currencies: List[str] = []
        errors: List[str] = list(document.errors)
        warnings: List[str] = list(document.warnings)

        if not chunks_to_process:
            msg = (
                f"No extractable text chunks found in document '{document.file_name}'. "
                "If this is a scanned document, ensure Tesseract OCR is installed and enabled."
            )
            logger.warning(msg)
            warnings.append(msg)

        for idx, chunk in enumerate(chunks_to_process):
            if progress_callback:
                progress_callback(
                    idx + 1,
                    total_to_process,
                    f"Processing page {chunk.page_number} ({idx+1}/{total_to_process})...",
                )

            resp, err = self.extract_from_chunk(chunk, model=resolved_model)
            if err:
                errors.append(err)
                continue

            if resp:
                if resp.company_name and resp.company_name.strip():
                    company_names.append(resp.company_name.strip())
                if resp.country and resp.country.strip():
                    countries.append(resp.country.strip())
                if resp.industry and resp.industry.strip():
                    industries.append(resp.industry.strip())
                if resp.reporting_period and resp.reporting_period.strip():
                    reporting_periods.append(resp.reporting_period.strip())
                if resp.financial_year:
                    financial_years.append(resp.financial_year)
                if resp.currency and resp.currency.strip():
                    currencies.append(resp.currency.strip())

                all_metrics.extend(resp.metrics)

        # Consensus resolution for company metadata
        def _pick_most_common(items: List[Any]) -> Optional[Any]:
            if not items:
                return None
            from collections import Counter
            counts = Counter(items)
            return counts.most_common(1)[0][0]

        resolved_company = _pick_most_common(company_names)
        resolved_country = _pick_most_common(countries)
        resolved_industry = _pick_most_common(industries)
        resolved_period = _pick_most_common(reporting_periods)
        resolved_year = _pick_most_common(financial_years)
        resolved_currency = _pick_most_common(currencies)

        # Propagate default currency/year to metrics if missing on the metric
        for m in all_metrics:
            if not m.year and resolved_year:
                m.year = resolved_year
            if not m.unit and resolved_currency:
                m.unit = resolved_currency
            if not m.reporting_period and resolved_period:
                m.reporting_period = resolved_period

        result = FinancialExtractionResult(
            document_name=document.file_name,
            company_name=resolved_company,
            country=resolved_country,
            industry=resolved_industry,
            reporting_period=resolved_period,
            financial_year=resolved_year,
            currency=resolved_currency,
            metrics=all_metrics,
            model_used=resolved_model,
            chunks_processed=total_to_process,
            total_chunks=len(all_chunks),
            extraction_warnings=warnings,
            extraction_errors=errors,
        )

        logger.info(
            "Extraction complete for '%s': %d metrics extracted, company='%s', year=%s",
            document.file_name,
            len(all_metrics),
            resolved_company,
            resolved_year,
        )
        return result

"""Page-aware chunking module preserving document provenance and page references."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from config import Settings, get_settings

logger = logging.getLogger("financial_intel.chunking")


@dataclass
class DocumentChunk:
    """A bounded segment of document text with complete source metadata."""

    chunk_id: str
    document_name: str
    page_number: int  # Primary 1-indexed page number
    page_range: List[int] = field(default_factory=list)
    text: str = ""
    char_count: int = 0
    token_estimate: int = 0
    source_types: List[str] = field(default_factory=list)  # e.g. ["text_layer", "table", "ocr"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert document chunk dataclass to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "page_range": self.page_range,
            "text": self.text,
            "char_count": self.char_count,
            "token_estimate": self.token_estimate,
            "source_types": self.source_types,
        }


class Chunker:
    """Splits normalized document content into manageable chunks for local LLM inference.

    Strictly preserves source page numbers, filenames, and table boundaries.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        cfg = settings or get_settings()
        self.chunk_size = chunk_size or cfg.chunk_size
        self.chunk_overlap = chunk_overlap or cfg.chunk_overlap
        self.max_chunks = cfg.max_chunks_per_document

        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 4)

    def chunk_page_content(
        self,
        page_text: str,
        page_number: int,
        document_name: str,
        source_type: str = "text_layer",
    ) -> List[DocumentChunk]:
        """Split a single page's text into one or more chunks preserving the page reference."""
        clean_text = page_text.strip()
        if not clean_text:
            return []

        # If page fits within chunk size, keep it whole (ideal for page traceability)
        if len(clean_text) <= self.chunk_size:
            return [
                DocumentChunk(
                    chunk_id=f"{document_name}_p{page_number}_c0",
                    document_name=document_name,
                    page_number=page_number,
                    page_range=[page_number],
                    text=clean_text,
                    char_count=len(clean_text),
                    token_estimate=max(1, len(clean_text) // 4),
                    source_types=[source_type],
                )
            ]

        # Multi-chunk splitting with overlap at paragraph boundaries
        paragraphs = clean_text.split("\n\n")
        chunks: List[DocumentChunk] = []
        current_paragraphs: List[str] = []
        current_length = 0
        chunk_idx = 0

        for para in paragraphs:
            para_len = len(para) + 2  # account for \n\n
            if current_length + para_len > self.chunk_size and current_paragraphs:
                chunk_body = "\n\n".join(current_paragraphs).strip()
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_name}_p{page_number}_c{chunk_idx}",
                        document_name=document_name,
                        page_number=page_number,
                        page_range=[page_number],
                        text=chunk_body,
                        char_count=len(chunk_body),
                        token_estimate=max(1, len(chunk_body) // 4),
                        source_types=[source_type],
                    )
                )
                chunk_idx += 1

                # Retain overlap from end of previous chunk if possible
                overlap_text = chunk_body[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                current_paragraphs = [overlap_text, para] if overlap_text else [para]
                current_length = len(overlap_text) + para_len
            else:
                current_paragraphs.append(para)
                current_length += para_len

        if current_paragraphs:
            chunk_body = "\n\n".join(current_paragraphs).strip()
            if chunk_body:
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_name}_p{page_number}_c{chunk_idx}",
                        document_name=document_name,
                        page_number=page_number,
                        page_range=[page_number],
                        text=chunk_body,
                        char_count=len(chunk_body),
                        token_estimate=max(1, len(chunk_body) // 4),
                        source_types=[source_type],
                    )
                )

        return chunks

    def chunk_document(
        self,
        pages_content: List[Dict[str, Any]],
        document_name: str,
    ) -> List[DocumentChunk]:
        """Process an entire document's pages into chunks, prioritizing financial relevance if exceeding max_chunks."""
        all_chunks: List[DocumentChunk] = []

        for p in pages_content:
            p_num = p.get("page_number", 1)
            p_text = p.get("text", "")
            p_src = p.get("source_type", "text_layer")
            page_chunks = self.chunk_page_content(
                page_text=p_text,
                page_number=p_num,
                document_name=document_name,
                source_type=p_src,
            )
            all_chunks.extend(page_chunks)

        # If chunks exceed max_chunks limit, prioritize chunks containing financial statements and tables
        if len(all_chunks) > self.max_chunks:
            logger.warning(
                "Document '%s' generated %d chunks exceeding limit of %d. Prioritizing financial content.",
                document_name,
                len(all_chunks),
                self.max_chunks,
            )
            from llm.financial_extractor import FINANCIAL_KEYWORDS

            def score_chunk(c: DocumentChunk) -> int:
                t = c.text.lower()
                score = 0
                if "[table" in t:
                    score += 5
                for kw in FINANCIAL_KEYWORDS:
                    if kw in t:
                        score += 1
                return score

            # Score all chunks
            scored_chunks = [(score_chunk(c), idx, c) for idx, c in enumerate(all_chunks)]
            # Sort by score descending, then by original index ascending
            scored_chunks.sort(key=lambda item: (-item[0], item[1]))
            # Take top max_chunks
            selected_items = scored_chunks[: self.max_chunks]
            # Restore original document reading order by index
            selected_items.sort(key=lambda item: item[1])
            all_chunks = [c for _, _, c in selected_items]

        logger.info(
            "Generated %d chunks for document '%s' (chunk_size=%d, overlap=%d)",
            len(all_chunks),
            document_name,
            self.chunk_size,
            self.chunk_overlap,
        )
        return all_chunks

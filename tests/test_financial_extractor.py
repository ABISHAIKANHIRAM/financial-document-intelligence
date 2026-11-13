"""Unit tests for Phase 3 local LLM financial extraction and JSON repair."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from core.chunking import DocumentChunk
from core.document_loader import PageContent, ProcessedDocument
from llm.financial_extractor import FinancialExtractor, repair_json_string
from llm.ollama_client import OllamaClient, OllamaConnectionError
from llm.prompts import build_chunk_extraction_prompt
from schemas.financial_schema import ExtractedMetric, FinancialExtractionResult


def test_repair_json_string_clean_json() -> None:
    raw = '{"company_name": "Test AG", "financial_year": 2024, "metrics": []}'
    repaired = repair_json_string(raw)
    assert repaired is not None
    assert repaired["company_name"] == "Test AG"
    assert repaired["financial_year"] == 2024


def test_repair_json_string_markdown_blocks() -> None:
    raw = """```json
    {
        "company_name": "Test AG",
        "currency": "EUR",
        "metrics": []
    }
    ```"""
    repaired = repair_json_string(raw)
    assert repaired is not None
    assert repaired["company_name"] == "Test AG"
    assert repaired["currency"] == "EUR"


def test_repair_json_string_with_conversational_text_and_trailing_commas() -> None:
    raw = """Here is the financial extraction you requested:
    {
        "company_name": "Global Corp",
        "financial_year": 2023,
        "metrics": [
            {
                "metric": "Revenue",
                "value": 15000000.0,
                "unit": "USD",
                "evidence": "Revenue was $15M",
                "source": {"page": 1, "document": "doc.pdf"},
            },
        ],
    }
    Hope this helps!"""
    repaired = repair_json_string(raw)
    assert repaired is not None
    assert repaired["company_name"] == "Global Corp"
    assert len(repaired["metrics"]) == 1
    assert repaired["metrics"][0]["value"] == 15000000.0


def test_repair_json_string_unrecoverable() -> None:
    assert repair_json_string("This is just plain text without any json brackets.") is None
    assert repair_json_string("") is None


def test_build_chunk_extraction_prompt_contains_provenance() -> None:
    prompt = build_chunk_extraction_prompt(
        chunk_text="Revenue reached EUR 10M.",
        document_name="annual_report.pdf",
        page_number=4,
        source_type="table",
    )
    assert "annual_report.pdf" in prompt
    assert "Page Number: 4" in prompt
    assert "Source Type: table" in prompt
    assert "Revenue reached EUR 10M." in prompt


def test_chunk_relevance_detection() -> None:
    extractor = FinancialExtractor()
    c1 = DocumentChunk(
        chunk_id="c1",
        document_name="doc.pdf",
        page_number=1,
        text="The company generated annual revenue of EUR 50M with EBITDA of EUR 12M.",
    )
    c2 = DocumentChunk(
        chunk_id="c2",
        document_name="doc.pdf",
        page_number=2,
        text="This page contains copyright notices and general legal terms and conditions.",
    )
    c3 = DocumentChunk(
        chunk_id="c3",
        document_name="doc.pdf",
        page_number=3,
        text="[TABLE p3_t1]\n| Metric | Value |\n| Total Assets | 100M |\n[/TABLE p3_t1]",
    )

    assert extractor.is_chunk_relevant(c1) is True
    assert extractor.is_chunk_relevant(c2) is False
    assert extractor.is_chunk_relevant(c3) is True


def test_extract_from_chunk_success() -> None:
    client = MagicMock(spec=OllamaClient)
    client.model = "qwen2.5:latest"
    llm_payload = {
        "company_name": "Solaris Energy AG",
        "financial_year": 2024,
        "currency": "EUR",
        "metrics": [
            {
                "metric": "Revenue",
                "value": 85000000.0,
                "unit": "EUR",
                "year": 2024,
                "evidence": "Total revenue amounted to EUR 85.0 million in 2024.",
                "source": {"page": 1, "document": "solaris.pdf"},
            }
        ],
    }
    client.generate.return_value = json.dumps(llm_payload)

    extractor = FinancialExtractor(ollama_client=client)
    chunk = DocumentChunk(
        chunk_id="solaris_p2_c0",
        document_name="solaris.pdf",
        page_number=2,
        text="Total revenue amounted to EUR 85.0 million in 2024.",
        source_types=["text_layer"],
    )

    resp, err = extractor.extract_from_chunk(chunk)
    assert err is None
    assert resp is not None
    assert resp.company_name == "Solaris Energy AG"
    assert len(resp.metrics) == 1
    # Verify strict source preservation matches the chunk's authentic page number
    assert resp.metrics[0].source.page == 2
    assert resp.metrics[0].source.document == "solaris.pdf"


def test_extract_from_chunk_retries_on_invalid_json() -> None:
    client = MagicMock(spec=OllamaClient)
    client.model = "qwen2.5:latest"
    # First response is malformed, second is valid
    valid_payload = {
        "company_name": "Fixed Corp",
        "financial_year": 2024,
        "currency": "USD",
        "metrics": [],
    }
    client.generate.side_effect = [
        "Sorry, I cannot produce json without error: {broken",
        json.dumps(valid_payload),
    ]

    extractor = FinancialExtractor(ollama_client=client)
    chunk = DocumentChunk(
        chunk_id="c0",
        document_name="test.pdf",
        page_number=1,
        text="Some text",
    )

    resp, err = extractor.extract_from_chunk(chunk, retry_on_invalid_json=True)
    assert err is None
    assert resp is not None
    assert resp.company_name == "Fixed Corp"
    assert client.generate.call_count == 2


def test_extract_from_document_aggregates_across_chunks() -> None:
    client = MagicMock(spec=OllamaClient)
    client.model = "qwen2.5:latest"
    client.ensure_ready.return_value = "qwen2.5:latest"

    # Chunk 1 returns company name and revenue
    p1 = {
        "company_name": "Bavaria Tech SE",
        "financial_year": 2024,
        "currency": "EUR",
        "metrics": [
            {
                "metric": "Revenue",
                "value": 120000000.0,
                "unit": "EUR",
                "year": 2024,
                "evidence": "Revenue hit EUR 120 million.",
                "source": {"page": 1, "document": "report.pdf"},
            }
        ],
    }
    # Chunk 2 returns EBITDA and Net Income
    p2 = {
        "company_name": "Bavaria Tech SE",
        "financial_year": 2024,
        "currency": "EUR",
        "metrics": [
            {
                "metric": "EBITDA",
                "value": 24000000.0,
                "unit": "EUR",
                "year": 2024,
                "evidence": "EBITDA increased to EUR 24 million.",
                "source": {"page": 2, "document": "report.pdf"},
            }
        ],
    }

    client.generate.side_effect = [json.dumps(p1), json.dumps(p2)]
    extractor = FinancialExtractor(ollama_client=client)

    doc = ProcessedDocument(
        file_path="report.pdf",
        file_name="report.pdf",
        file_type="pdf",
        file_size_bytes=1000,
        page_count=2,
        pages=[
            PageContent(page_number=1, text="Revenue hit EUR 120 million."),
            PageContent(page_number=2, text="EBITDA increased to EUR 24 million."),
        ],
        chunks=[
            DocumentChunk(
                chunk_id="c1",
                document_name="report.pdf",
                page_number=1,
                text="Revenue hit EUR 120 million.",
            ),
            DocumentChunk(
                chunk_id="c2",
                document_name="report.pdf",
                page_number=2,
                text="EBITDA increased to EUR 24 million.",
            ),
        ],
    )

    result = extractor.extract_from_document(doc)

    assert result.company_name == "Bavaria Tech SE"
    assert result.financial_year == 2024
    assert result.currency == "EUR"
    assert len(result.metrics) == 2
    assert result.metrics[0].metric == "Revenue"
    assert result.metrics[0].source.page == 1
    assert result.metrics[1].metric == "EBITDA"
    assert result.metrics[1].source.page == 2
    assert len(result.extraction_errors) == 0


def test_extract_from_document_handles_ollama_offline() -> None:
    client = MagicMock(spec=OllamaClient)
    client.model = "qwen2.5:latest"
    client.ensure_ready.side_effect = OllamaConnectionError("Ollama is not running")

    extractor = FinancialExtractor(ollama_client=client)
    doc = ProcessedDocument(
        file_path="test.pdf",
        file_name="test.pdf",
        file_type="pdf",
        file_size_bytes=100,
        page_count=1,
    )

    result = extractor.extract_from_document(doc)
    assert len(result.metrics) == 0
    assert len(result.extraction_errors) == 1
    assert "Ollama is not running" in result.extraction_errors[0]

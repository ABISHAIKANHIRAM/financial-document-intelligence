"""Structured prompts for local LLM financial extraction and source verification."""

from __future__ import annotations

import json
from typing import Optional

FINANCIAL_EXTRACTION_SYSTEM_PROMPT = """You are a senior financial analyst and structured data extraction engine.
Your mission is to extract company metadata and financial metrics with 100% precision and source traceability from the provided text excerpt.

### STRICT RULES:
1. OUTPUT FORMAT: Respond ONLY with a valid JSON object matching the exact schema below. Do NOT include markdown code blocks (```json), explanations, or conversational filler.
2. SOURCE TRACEABILITY: Every metric MUST include the verbatim evidence sentence from the document, along with the provided document name and page number.
3. NEVER HALLUCINATE OR GUESS: If a metric or company attribute is not explicitly mentioned in the text, return null or omit it. Never invent numbers.
4. FINANCIAL DISTINCTIONS:
   - Revenue / Turnover: Top-line sales before any deductions.
   - Operating Profit / EBIT: Operating earnings before interest and taxes.
   - EBITDA: Operating earnings before interest, taxes, depreciation, and amortization. Never confuse EBITDA with Revenue or Net Income!
   - Net Income / Net Profit: Bottom-line profit or loss after taxes and interest.
   - Total Assets vs Total Liabilities: Do not confuse what the company owns with what it owes.
   - Cash Flow: Cash flow from operating, investing, or financing activities.
   - Number of Employees: Full-time employees or headcount.
   - Growth Percentage: Year-over-year or period percentage growth (unit MUST be "%").
5. UNITS & SCALING:
   - If values are expressed in millions (e.g., "EUR 12.5 million" or table header "in millions EUR"), convert the numeric value to the full absolute number (12500000) and set unit to "EUR".
   - If expressed in thousands (e.g., "$'000"), multiply by 1000 (e.g., 450 -> 450000).
   - Distinguish currencies (EUR, USD, GBP, etc.). Never confuse EUR with USD.
6. NEGATIVE VALUES: Preserve negative amounts (e.g. accounting brackets "(1,250)" or "net loss of $2M" must be -2000000).

### TARGET JSON SCHEMA:
{
  "company_name": "string or null",
  "country": "string or null",
  "industry": "string or null",
  "reporting_period": "string or null (e.g. FY 2024, Q3 2024)",
  "financial_year": "integer or null (e.g. 2024)",
  "currency": "string or null (e.g. EUR, USD)",
  "metrics": [
    {
      "metric": "Revenue | EBITDA | EBIT | Operating Profit | Net Income | Total Assets | Total Liabilities | Cash Flow | Number of Employees | Growth Percentage",
      "value": 12500000.0,
      "unit": "EUR | USD | % | count",
      "year": 2024,
      "reporting_period": "FY 2024",
      "evidence": "Exact verbatim sentence from text supporting this value.",
      "source": {
        "page": 1,
        "document": "document_name.pdf",
        "source_type": "text_layer"
      },
      "extraction_method": "llm_extraction",
      "verification_status": "pending"
    }
  ]
}
"""


def build_chunk_extraction_prompt(
    chunk_text: str,
    document_name: str,
    page_number: int,
    source_type: str = "text_layer",
) -> str:
    """Build a structured extraction prompt for a single document chunk."""
    return f"""Analyze the following financial document excerpt and extract all company metadata and financial metrics according to your instructions.

DOCUMENT PROVENANCE:
- Document Name: {document_name}
- Page Number: {page_number}
- Source Type: {source_type}

--- DOCUMENT EXCERPT START ---
{chunk_text}
--- DOCUMENT EXCERPT END ---

Return strictly a valid JSON object matching the target schema.
Ensure every metric's source.page is {page_number} and source.document is "{document_name}".
"""


def build_corrective_repair_prompt(
    raw_response: str,
    error_message: str,
    target_schema_example: Optional[str] = None,
) -> str:
    """Build a corrective prompt instructing the local LLM to fix syntax or schema errors."""
    schema_hint = target_schema_example or "JSON object with company_name, financial_year, currency, and metrics list."
    return f"""The previous response could not be parsed as valid JSON or failed validation.

ERROR:
{error_message}

PREVIOUS INVALID RESPONSE:
{raw_response}

REQUIRED SCHEMA:
{schema_hint}

INSTRUCTION:
Fix all formatting, syntax, and schema errors. Output ONLY the corrected, clean, raw JSON object.
Do not include any commentary or markdown code fences.
"""

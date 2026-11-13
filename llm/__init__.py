"""Local LLM integration modules (Ollama client, prompts, financial extractor)."""

from llm.financial_extractor import FinancialExtractor, repair_json_string
from llm.ollama_client import (
    InstalledModel,
    OllamaClient,
    OllamaConnectionError,
    OllamaError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaStatus,
    OllamaTimeoutError,
)
from llm.prompts import (
    FINANCIAL_EXTRACTION_SYSTEM_PROMPT,
    build_chunk_extraction_prompt,
    build_corrective_repair_prompt,
)

__all__ = [
    "FINANCIAL_EXTRACTION_SYSTEM_PROMPT",
    "FinancialExtractor",
    "InstalledModel",
    "OllamaClient",
    "OllamaConnectionError",
    "OllamaError",
    "OllamaModelNotFoundError",
    "OllamaResponseError",
    "OllamaStatus",
    "OllamaTimeoutError",
    "build_chunk_extraction_prompt",
    "build_corrective_repair_prompt",
    "repair_json_string",
]

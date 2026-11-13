"""Unit tests for Phase 1 configuration and local Ollama client."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import requests

from config import Settings
from llm.ollama_client import (
    InstalledModel,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
)


def _mock_response(status_code: int = 200, json_data=None, text: str = "") -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("No JSON")
    if status_code >= 400:
        http_err = requests.HTTPError(f"HTTP {status_code}")
        resp.raise_for_status.side_effect = http_err
    else:
        resp.raise_for_status.return_value = None
    return resp


def test_settings_rejects_non_local_ollama_by_default() -> None:
    settings = Settings(ollama_base_url="https://api.example.com:11434")
    with pytest.raises(ValueError, match="Refusing non-local Ollama endpoint"):
        settings.ensure_local_ollama()


def test_settings_allows_localhost_variants() -> None:
    for url in ("http://localhost:11434", "http://127.0.0.1:11434"):
        settings = Settings(ollama_base_url=url)
        settings.ensure_local_ollama()


def test_ollama_client_blocks_external_host() -> None:
    with pytest.raises(OllamaConnectionError, match="Blocked non-local Ollama URL"):
        OllamaClient(base_url="http://remote-server.internal:11434", allow_non_local=False)


def test_list_models_and_resolve_tags() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response(
        200,
        {
            "models": [
                {
                    "name": "qwen2.5:7b",
                    "size": 4_700_000_000,
                    "modified_at": "2026-01-01T00:00:00Z",
                    "details": {
                        "family": "qwen2",
                        "parameter_size": "7.6B",
                        "quantization_level": "Q4_K_M",
                    },
                },
                {"name": "llama3.1:latest", "size": 4_900_000_000},
            ]
        },
    )
    client = OllamaClient(model="qwen2.5", session=session)
    status = client.get_status()

    assert status.is_running is True
    assert status.model_available is True
    assert status.resolved_model == "qwen2.5:7b"
    assert len(status.installed_models) == 2


def test_resolve_model_name_prefers_latest_for_untagged() -> None:
    models = [
        InstalledModel(name="qwen2.5:14b"),
        InstalledModel(name="qwen2.5:latest"),
    ]
    assert OllamaClient.resolve_model_name("qwen2.5", models) == "qwen2.5:latest"
    assert OllamaClient.resolve_model_name("qwen2.5:14b", models) == "qwen2.5:14b"
    assert OllamaClient.resolve_model_name("qwen2.5:32b", models) is None


def test_status_reports_missing_model_cleanly() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response(
        200, {"models": [{"name": "mistral:latest", "size": 4_000_000_000}]}
    )
    client = OllamaClient(model="qwen2.5:latest", session=session)
    status = client.get_status()

    assert status.is_running is True
    assert status.model_available is False
    assert status.resolved_model is None
    assert "ollama pull qwen2.5:latest" in (status.error_message or "")


def test_status_reports_connection_failure() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = requests.ConnectionError("Connection refused")
    client = OllamaClient(model="qwen2.5:latest", session=session)
    status = client.get_status()

    assert status.is_running is False
    assert status.model_available is False
    assert "Could not connect to local Ollama server" in (status.error_message or "")


def test_generate_success_uses_resolved_model() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response(
        200, {"models": [{"name": "qwen2.5:latest", "size": 4_500_000_000}]}
    )
    session.post.return_value = _mock_response(
        200, {"response": '{"company_name": "Example GmbH"}'}
    )
    client = OllamaClient(model="qwen2.5", temperature=0.0, num_ctx=4096, session=session)
    output = client.generate("Extract company name", response_format="json")

    assert output == '{"company_name": "Example GmbH"}'
    called_payload = session.post.call_args.kwargs["json"]
    assert called_payload["model"] == "qwen2.5:latest"
    assert called_payload["format"] == "json"
    assert called_payload["options"]["temperature"] == 0.0
    assert called_payload["options"]["num_ctx"] == 4096


def test_generate_timeout_raises_ollama_timeout_error() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response(
        200, {"models": [{"name": "qwen2.5:latest", "size": 4_500_000_000}]}
    )
    session.post.side_effect = requests.ReadTimeout("Timed out")
    client = OllamaClient(model="qwen2.5:latest", timeout_seconds=10, session=session)

    with pytest.raises(OllamaTimeoutError, match="timed out after 10s"):
        client.generate("Hello")


def test_generate_missing_model_raises_not_found() -> None:
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _mock_response(200, {"models": []})
    client = OllamaClient(model="qwen2.5:latest", session=session)

    with pytest.raises(OllamaModelNotFoundError):
        client.generate("Hello")


def test_chat_malformed_response_raises_response_error() -> None:
    session = MagicMock(spec=requests.Session)
    session.post.return_value = _mock_response(200, {"unexpected": "shape"})
    client = OllamaClient(model="qwen2.5:latest", session=session)

    with pytest.raises(OllamaResponseError, match="missing 'message.content'"):
        client.chat([{"role": "user", "content": "Hi"}], verify_model=False)

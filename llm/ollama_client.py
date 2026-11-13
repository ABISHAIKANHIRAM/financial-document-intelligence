"""Local Ollama HTTP client for privacy-preserving LLM inference."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from requests import RequestException, Timeout

from config import LOCAL_HOSTNAMES, Settings, get_settings

logger = logging.getLogger("financial_intel.ollama")


class OllamaError(Exception):
    """Base exception for local Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Raised when the local Ollama server cannot be reached or is non-local."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested Ollama model is not installed locally."""


class OllamaTimeoutError(OllamaError):
    """Raised when a local Ollama inference request times out."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an unexpected HTTP status or malformed payload."""


@dataclass(frozen=True)
class InstalledModel:
    """Metadata describing a locally installed Ollama model."""

    name: str
    size_bytes: int = 0
    modified_at: str = ""
    family: str = ""
    parameter_size: str = ""
    quantization_level: str = ""

    @property
    def size_gb(self) -> float:
        """Return model size in gigabytes rounded to two decimals."""
        return round(self.size_bytes / (1024**3), 2) if self.size_bytes > 0 else 0.0


@dataclass(frozen=True)
class OllamaStatus:
    """Snapshot of local Ollama server health and model readiness."""

    is_running: bool
    base_url: str
    configured_model: str
    resolved_model: Optional[str] = None
    model_available: bool = False
    installed_models: List[InstalledModel] = field(default_factory=list)
    error_message: Optional[str] = None


class OllamaClient:
    """HTTP client for interacting strictly with a local Ollama instance."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        allow_non_local: Optional[bool] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        settings: Settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).strip().rstrip("/")
        self.model = (model or settings.ollama_model).strip()
        self.temperature = (
            settings.ollama_temperature if temperature is None else temperature
        )
        self.num_ctx = settings.ollama_num_ctx if num_ctx is None else num_ctx
        self.timeout_seconds = (
            settings.ollama_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        self.allow_non_local = (
            settings.allow_non_local_ollama
            if allow_non_local is None
            else allow_non_local
        )
        self._session = session or requests.Session()
        self._validate_local_endpoint()

    def _validate_local_endpoint(self) -> None:
        """Prevent sending financial documents to external hosts."""
        parsed = urlparse(self.base_url)
        hostname = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or not hostname:
            raise OllamaConnectionError(
                f"Invalid Ollama base URL '{self.base_url}'. Expected e.g. http://localhost:11434."
            )
        if not self.allow_non_local and hostname not in LOCAL_HOSTNAMES:
            raise OllamaConnectionError(
                f"Blocked non-local Ollama URL '{self.base_url}'. "
                "Financial documents must only be processed locally."
            )

    def _get_json(self, path: str, timeout: int = 5) -> Dict[str, Any]:
        """Execute a GET request against Ollama and return parsed JSON."""
        url = f"{self.base_url}{path}"
        try:
            response = self._session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Timeout as exc:
            raise OllamaTimeoutError(
                f"Timed out connecting to Ollama at {url} after {timeout}s."
            ) from exc
        except RequestException as exc:
            raise OllamaConnectionError(
                f"Could not connect to local Ollama server at {self.base_url}. "
                "Ensure Ollama is installed and running (`ollama serve`)."
            ) from exc
        except ValueError as exc:
            raise OllamaResponseError(
                f"Ollama returned invalid JSON from {url}."
            ) from exc

    def _post_json(
        self, path: str, payload: Dict[str, Any], timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute a POST request against Ollama and return parsed JSON."""
        url = f"{self.base_url}{path}"
        effective_timeout = timeout or self.timeout_seconds
        try:
            response = self._session.post(url, json=payload, timeout=effective_timeout)
        except Timeout as exc:
            raise OllamaTimeoutError(
                f"Ollama request to {path} timed out after {effective_timeout}s."
            ) from exc
        except RequestException as exc:
            raise OllamaConnectionError(
                f"Failed communicating with local Ollama server at {self.base_url}: {exc}"
            ) from exc

        if response.status_code == 404:
            detail = self._extract_error_detail(response)
            raise OllamaModelNotFoundError(
                detail
                or f"Ollama endpoint or model not found ({path}). Pull the model with `ollama pull {payload.get('model', self.model)}`."
            )

        if response.status_code >= 400:
            detail = self._extract_error_detail(response)
            raise OllamaResponseError(
                f"Ollama returned HTTP {response.status_code}: {detail or response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaResponseError(
                f"Ollama returned malformed JSON from {path}."
            ) from exc

        if "error" in data and data["error"]:
            raise OllamaResponseError(f"Ollama error: {data['error']}")
        return data

    @staticmethod
    def _extract_error_detail(response: requests.Response) -> str:
        """Extract an error message from an Ollama error response body."""
        try:
            body = response.json()
            if isinstance(body, dict) and "error" in body:
                return str(body["error"])
        except ValueError:
            pass
        return response.text.strip()

    def is_running(self) -> bool:
        """Return True if the local Ollama server responds cleanly."""
        try:
            self._get_json("/api/tags", timeout=5)
            return True
        except OllamaError:
            return False

    def list_models(self) -> List[InstalledModel]:
        """Return metadata for all models currently installed in Ollama."""
        payload = self._get_json("/api/tags", timeout=10)
        raw_models = payload.get("models", [])
        installed: List[InstalledModel] = []
        for item in raw_models:
            details = item.get("details") or {}
            installed.append(
                InstalledModel(
                    name=str(item.get("name", "")),
                    size_bytes=int(item.get("size") or 0),
                    modified_at=str(item.get("modified_at") or ""),
                    family=str(details.get("family") or ""),
                    parameter_size=str(details.get("parameter_size") or ""),
                    quantization_level=str(details.get("quantization_level") or ""),
                )
            )
        return sorted(installed, key=lambda m: m.name.lower())

    @staticmethod
    def resolve_model_name(
        requested_model: str, installed_models: List[InstalledModel]
    ) -> Optional[str]:
        """Match a requested model name against locally installed Ollama tags."""
        target = requested_model.strip()
        if not target:
            return None

        installed_names = [m.name for m in installed_models]
        if target in installed_names:
            return target

        # If no tag was specified (e.g. 'qwen2.5'), prefer ':latest' then any matching family tag.
        if ":" not in target:
            latest_candidate = f"{target}:latest"
            if latest_candidate in installed_names:
                return latest_candidate
            prefix = f"{target}:"
            for name in installed_names:
                if name.startswith(prefix):
                    return name
        return None

    def is_model_available(self, model_name: Optional[str] = None) -> bool:
        """Return True if the configured or specified model is installed locally."""
        target = model_name or self.model
        installed = self.list_models()
        return self.resolve_model_name(target, installed) is not None

    def get_status(self, model_name: Optional[str] = None) -> OllamaStatus:
        """Inspect Ollama connectivity, installed models, and configured model availability."""
        target_model = (model_name or self.model).strip()
        try:
            installed = self.list_models()
        except OllamaError as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return OllamaStatus(
                is_running=False,
                base_url=self.base_url,
                configured_model=target_model,
                error_message=str(exc),
            )

        resolved = self.resolve_model_name(target_model, installed)
        if resolved is None:
            available_names = ", ".join(m.name for m in installed) or "none"
            err = (
                f"Configured model '{target_model}' is not installed locally. "
                f"Installed models: {available_names}. "
                f"Run `ollama pull {target_model}` to install it."
            )
            logger.warning(err)
            return OllamaStatus(
                is_running=True,
                base_url=self.base_url,
                configured_model=target_model,
                resolved_model=None,
                model_available=False,
                installed_models=installed,
                error_message=err,
            )

        return OllamaStatus(
            is_running=True,
            base_url=self.base_url,
            configured_model=target_model,
            resolved_model=resolved,
            model_available=True,
            installed_models=installed,
            error_message=None,
        )

    def ensure_ready(self, model_name: Optional[str] = None) -> str:
        """Verify Ollama is running and the target model is available; return resolved name."""
        status = self.get_status(model_name=model_name)
        if not status.is_running:
            raise OllamaConnectionError(
                status.error_message
                or f"Ollama is not running at {self.base_url}."
            )
        if not status.model_available or not status.resolved_model:
            raise OllamaModelNotFoundError(
                status.error_message
                or f"Model '{status.configured_model}' is not installed locally."
            )
        return status.resolved_model

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        response_format: Optional[str] = None,
        verify_model: bool = True,
    ) -> str:
        """Run a single-turn completion against `/api/generate` and return response text."""
        target_model = model or self.model
        if verify_model:
            target_model = self.ensure_ready(target_model)

        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": (
                    self.temperature if temperature is None else temperature
                ),
                "num_ctx": self.num_ctx if num_ctx is None else num_ctx,
            },
        }
        if system:
            payload["system"] = system
        if response_format:
            payload["format"] = response_format

        logger.debug(
            "Calling Ollama /api/generate with model=%s, num_ctx=%s",
            target_model,
            payload["options"]["num_ctx"],
        )
        data = self._post_json("/api/generate", payload)
        text = data.get("response")
        if text is None:
            raise OllamaResponseError(
                "Ollama /api/generate response missing 'response' field."
            )
        return str(text)

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
        response_format: Optional[str] = None,
        verify_model: bool = True,
    ) -> str:
        """Run a multi-message completion against `/api/chat` and return assistant text."""
        target_model = model or self.model
        if verify_model:
            target_model = self.ensure_ready(target_model)

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": (
                    self.temperature if temperature is None else temperature
                ),
                "num_ctx": self.num_ctx if num_ctx is None else num_ctx,
            },
        }
        if response_format:
            payload["format"] = response_format

        logger.debug(
            "Calling Ollama /api/chat with model=%s, messages=%d",
            target_model,
            len(messages),
        )
        data = self._post_json("/api/chat", payload)
        message = data.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise OllamaResponseError(
                "Ollama /api/chat response missing 'message.content'."
            )
        return str(message["content"])


def main() -> int:
    """CLI entry point to verify local Ollama connectivity and model availability."""
    from config import configure_logging

    settings = get_settings()
    configure_logging(settings)
    client = OllamaClient()
    status = client.get_status()

    summary = {
        "ollama_running": status.is_running,
        "base_url": status.base_url,
        "configured_model": status.configured_model,
        "resolved_model": status.resolved_model,
        "model_available": status.model_available,
        "installed_models": [
            {
                "name": m.name,
                "size_gb": m.size_gb,
                "parameter_size": m.parameter_size,
            }
            for m in status.installed_models
        ],
        "error_message": status.error_message,
    }
    print(json.dumps(summary, indent=2))
    return 0 if (status.is_running and status.model_available) else 1


if __name__ == "__main__":
    raise SystemExit(main())

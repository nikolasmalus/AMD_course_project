from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import requests


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def is_local_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.hostname in LOCAL_HOSTS


class LocalLLMClient:
    """Local-only Ollama client for event alert generation."""

    def __init__(
        self,
        api_url: str,
        model_name: str,
        timeout_seconds: int = 20,
        allow_external_api: bool = False,
    ) -> None:
        self.api_url = api_url
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.allow_external_api = allow_external_api
        self.local_only = is_local_url(api_url)
        if not self.local_only and not allow_external_api:
            raise ValueError("Only local Ollama API URLs are allowed: 127.0.0.1, localhost, or ::1.")

    def _metadata(self, llm_used: bool, error: str | None = None) -> dict[str, Any]:
        return {
            "llm_used": llm_used,
            "provider": "ollama",
            "model_name": self.model_name,
            "local_only": self.local_only,
            "error": error,
        }

    def is_available(self) -> tuple[bool, str | None]:
        text, metadata = self.generate("请回复：ok")
        if text and metadata.get("llm_used"):
            return True, None
        return False, metadata.get("error") or "Ollama returned no response."

    def generate(self, prompt: str) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            text = str(data.get("response", "")).strip()
            if not text:
                return "", self._metadata(False, "Ollama returned an empty response.")
            return text, self._metadata(True, None)
        except Exception as exc:
            return "", self._metadata(False, f"Ollama request failed: {exc}")

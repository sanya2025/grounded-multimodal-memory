"""Tiny stdlib HTTP client for a local Ollama server.

No third-party dependencies (uses urllib) so the base package stays light. Both
the Ollama VLM adapter and the Ollama embedder call ``post_json`` here. Tests
patch ``urlopen`` in this module to avoid needing a running server.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen


def default_endpoint() -> str:
    """Ollama base URL: $OLLAMA_HOST if set, else http://localhost:11434."""
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def post_json(
    path: str,
    payload: dict[str, Any],
    endpoint: str | None = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """POST ``payload`` as JSON to ``endpoint + path`` and return the parsed JSON.

    Raises a clear ``RuntimeError`` if the server can't be reached (the most
    common cause is that ``ollama serve`` isn't running).
    """
    base = (endpoint or default_endpoint()).rstrip("/")
    url = f"{base}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:  # noqa: S310 (local, trusted URL)
            return json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {url}: {exc}. "
            f"Is the Ollama app / `ollama serve` running, and the model pulled?"
        ) from exc


__all__ = ["default_endpoint", "post_json"]

"""Client for a local OpenAI-compatible model server (LM Studio, Ollama, or vLLM in production).

Ported from the local-rag module in secure-dev-platform, keeping its two safety properties:
the endpoint must be on this machine (claims data never leaves it), and every caller has an
offline path, so the pipeline and API still work when no model server is running.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import numpy as np

DEFAULT_BASE = "http://127.0.0.1:1234/v1"
# Loopback, plus Docker Desktop's name for the host machine: still the same computer.
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "host.docker.internal"}
# A 3B model answers in seconds on a laptop CPU; larger models stay available via LLM_CHAT_MODEL.
CHAT_PREFERENCE = ["qwen2.5-3b-instruct", "qwen2.5-7b-instruct", "llama-3.2-3b-instruct"]


class LLMUnavailable(RuntimeError):
    """No local model server, or the needed model is not loaded."""


def assert_local(base: str) -> str:
    parsed = urlparse(base)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported LLM URL scheme: {parsed.scheme!r}")
    if (parsed.hostname or "") not in LOCAL_HOSTS:
        raise ValueError(f"Refusing non-local inference host: {parsed.hostname}. Claims data must stay on this machine.")
    return base.rstrip("/")


def llm_disabled() -> bool:
    return os.environ.get("FRAUDOPS_NO_LLM", "").strip() not in ("", "0", "false")


@dataclass
class ModelChoice:
    chat: str | None
    embed: str | None


class LocalLLM:
    def __init__(self, base: str | None = None):
        self.base = assert_local(base or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE)
        self._choice: ModelChoice | None = None

    def _request(self, method: str, path: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
        if llm_disabled():
            raise LLMUnavailable("LLM calls are disabled (FRAUDOPS_NO_LLM)")
        try:
            response = httpx.request(method, self.base + path, json=payload, timeout=timeout, follow_redirects=False)
        except httpx.HTTPError as error:
            raise LLMUnavailable(f"Local model server not reachable at {self.base}: {error.__class__.__name__}") from error
        if response.status_code != 200:
            raise LLMUnavailable(f"Local model server returned HTTP {response.status_code}")
        return response.json()

    def models(self) -> list[str]:
        data = self._request("GET", "/models", timeout=3.0)
        return [item["id"] for item in data.get("data", []) if isinstance(item.get("id"), str)]

    def choose(self) -> ModelChoice:
        if self._choice is None:
            ids = self.models()
            embed = next((i for i in ids if "embed" in i.lower()), None)
            override = os.environ.get("LLM_CHAT_MODEL")
            chat = (override if override in ids else None) \
                or next((p for p in CHAT_PREFERENCE if p in ids), None) \
                or next((i for i in ids if "embed" not in i.lower()), None)
            self._choice = ModelChoice(chat=chat, embed=embed)
        return self._choice

    def embed(self, texts: list[str], kind: str = "document") -> np.ndarray:
        """Unit-normalised embeddings. Nomic models expect asymmetric task prefixes."""
        model = self.choose().embed
        if not model:
            raise LLMUnavailable("No embedding model loaded")
        prefix = "search_query: " if kind == "query" else "search_document: "
        vectors = []
        for start in range(0, len(texts), 16):
            batch = [prefix + text for text in texts[start:start + 16]]
            data = self._request("POST", "/embeddings", {"model": model, "input": batch}, timeout=120.0)
            rows = sorted(data.get("data", []), key=lambda row: row["index"])
            if len(rows) != len(batch):
                raise LLMUnavailable("Embedding count mismatch")
            vectors += [row["embedding"] for row in rows]
        matrix = np.asarray(vectors, dtype=np.float32)
        return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)

    def chat(self, system: str, user: str, max_tokens: int = 450, temperature: float = 0.2) -> str:
        model = self.choose().chat
        if not model:
            raise LLMUnavailable("No chat model loaded")
        data = self._request("POST", "/chat/completions", {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }, timeout=240.0)
        answer = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        # Reasoning models may wrap their thinking in tags; only the answer is shown.
        answer = answer.split("</think>")[-1].replace("<think>", "").strip()
        if not answer:
            raise LLMUnavailable("The model returned an empty answer")
        return answer

    def status(self) -> dict:
        status = {"endpoint": self.base, "local_only": True, "chat_model": None, "embed_model": None,
                  "ready": False, "reason": None}
        try:
            choice = self.choose()
            status.update(chat_model=choice.chat, embed_model=choice.embed, ready=bool(choice.chat and choice.embed))
            if not status["ready"]:
                status["reason"] = "Load a chat and an embedding model in the local model server"
        except LLMUnavailable as error:
            status["reason"] = str(error)
        return status

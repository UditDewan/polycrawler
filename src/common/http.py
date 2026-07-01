"""Shared networking: a polite HTTP getter for collectors and an OpenAI-compatible
client factory — both over the OS cert store so TLS-intercepting proxies validate."""
from __future__ import annotations

import os
import time

import httpx

try:  # use the OS cert store so corporate/TLS-intercepting proxies validate
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - truststore optional at runtime
    pass

USER_AGENT = "polycrawler/0.1 (research; +https://github.com/)"


def openai_client(base_url: str, api_key_env: str = "NVIDIA_API_KEY", *, timeout: float = 90.0):
    """OpenAI-compatible client for the hosted endpoint. Raises ValueError if the key
    env var is unset (callers wrap it in their own LLM/Embeddings 'unavailable' error).
    `timeout` caps each request so a slow/hung generation fails instead of blocking forever."""
    from openai import OpenAI  # lazy: only needed for real calls, not for tests

    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} not set")
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def get(url: str, *, retries: int = 3, timeout: float = 30.0, headers: dict | None = None) -> httpx.Response:
    h = {"User-Agent": USER_AGENT}
    h.update(headers or {})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = httpx.get(url, headers=h, timeout=timeout, follow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001 - retry any transport/HTTP error
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # 1s, 2s backoff
    raise last  # type: ignore[misc]

"""One polite HTTP getter for all collectors: OS-trust certs, a descriptive
User-Agent, and retry-with-backoff. Nothing fancy — collectors call get()."""
from __future__ import annotations

import time

import httpx

try:  # use the OS cert store so corporate/TLS-intercepting proxies validate
    import truststore

    truststore.inject_into_ssl()
except Exception:  # pragma: no cover - truststore optional at runtime
    pass

USER_AGENT = "polycrawler/0.1 (research; +https://github.com/)"


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

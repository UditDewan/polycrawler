"""Provider-agnostic LLM client (OpenAI-compatible).

Responsibilities the rest of the codebase should never re-implement:
  - response caching keyed on (model, params, prompt) — never pay twice for the same text
  - transport retry with backoff; raise LLMUnavailable on exhaustion (caller stops + resumes)
  - strict JSON: validate against a Pydantic schema, corrective-retry on invalid output,
    return None (quarantine) if the model never complies

Provider + model come from config. Inject `client=` (anything exposing
`chat.completions.create`) to unit-test without network or an API key.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from . import http

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """API unreachable or quota exhausted after retries — stop the batch and resume later."""


class LLMClient:
    def __init__(self, *, model, cache_dir, client, temperature=0.0, max_tokens=1024, max_retries=3):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = client
        self._cache = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, cfg, *, client=None):
        llm = cfg["llm"]
        return cls(
            model=llm["extract_model"],
            cache_dir=llm.get("cache_dir", "data/llm_cache"),
            client=client if client is not None else _default_client(llm),
            temperature=llm.get("temperature", 0.0),
            max_tokens=llm.get("max_tokens", 1024),
            max_retries=llm.get("max_retries", 3),
        )

    def _key(self, system: str, user: str) -> str:
        h = hashlib.sha256()
        for part in (self.model, str(self.temperature), str(self.max_tokens), system, user):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def complete_json(self, system: str, user: str, *, schema: type[T]) -> T | None:
        """Validated `schema` instance, or None if the model never produced valid JSON.
        Raises LLMUnavailable on API failure (so the caller can stop and resume)."""
        path = self._cache / (self._key(system, user) + ".json")
        if path.exists():
            try:
                return schema.model_validate_json(path.read_text(encoding="utf-8"))
            except ValidationError:
                path.unlink(missing_ok=True)  # stale cache from an older schema; refetch

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(self.max_retries):
            text = self._call(messages)  # may raise LLMUnavailable
            try:
                obj = schema.model_validate_json(text)
            except ValidationError as e:
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content":
                        f"That response had {e.error_count()} schema error(s). "
                        "Return ONLY a JSON object matching the schema — no prose, no markdown."},
                ]
                continue
            path.write_text(obj.model_dump_json(), encoding="utf-8")
            return obj
        return None  # quarantine

    def complete_text(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        """Plain-text completion (no JSON mode) — for summaries / digests."""
        return self._call([{"role": "system", "content": system},
                           {"role": "user", "content": user}],
                          json_mode=False, max_tokens=max_tokens)

    def _call(self, messages, *, json_mode: bool = True, max_tokens: int | None = None) -> str:
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature,
                      max_tokens=max_tokens or self.max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._client.chat.completions.create(**kwargs).choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - retry any transport/quota error
                last = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        raise LLMUnavailable(repr(last))


def _default_client(llm: dict):
    try:
        return http.openai_client(llm["base_url"], llm.get("api_key_env", "NVIDIA_API_KEY"))
    except ValueError as e:
        raise LLMUnavailable(str(e))

"""Provider-agnostic embeddings (OpenAI-compatible).

Per-text caching by input hash, batched misses, and NVIDIA's required `input_type`
('passage' for indexed docs, 'query' for searches). Inject `client=` to test
without network. Mirrors llm_client.py — same caching/quota discipline.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import http


class EmbeddingsUnavailable(RuntimeError):
    """Embedding API unreachable or quota exhausted."""


class EmbeddingsClient:
    def __init__(self, *, model, dim, cache_dir, client, batch_size=64, truncate="END"):
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.truncate = truncate
        self._client = client
        self._cache = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, cfg, *, client=None):
        emb = cfg["embeddings"]
        return cls(
            model=emb["model"], dim=emb["dim"],
            cache_dir=emb.get("cache_dir", "data/emb_cache"),
            client=client if client is not None else _default_client(emb),
            batch_size=emb.get("batch_size", 64), truncate=emb.get("truncate", "END"),
        )

    def _key(self, text: str, input_type: str) -> str:
        h = hashlib.sha256()
        for part in (self.model, input_type, text):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def embed(self, texts: list[str], *, input_type: str) -> list[list[float]]:
        """Embed texts (cached per text). input_type is 'passage' or 'query'."""
        out: list[list[float] | None] = [None] * len(texts)
        misses, miss_idx = [], []
        for i, t in enumerate(texts):
            p = self._cache / (self._key(t, input_type) + ".json")
            if p.exists():
                out[i] = json.loads(p.read_text(encoding="utf-8"))
            else:
                misses.append(t)
                miss_idx.append(i)
        for start in range(0, len(misses), self.batch_size):
            batch = misses[start:start + self.batch_size]
            for j, vec in enumerate(self._call(batch, input_type)):
                i = miss_idx[start + j]
                out[i] = vec
                (self._cache / (self._key(batch[j], input_type) + ".json")).write_text(
                    json.dumps(vec), encoding="utf-8")
        return out  # type: ignore[return-value]

    def embed_one(self, text: str, *, input_type: str) -> list[float]:
        return self.embed([text], input_type=input_type)[0]

    def _call(self, batch: list[str], input_type: str) -> list[list[float]]:
        try:
            resp = self._client.embeddings.create(
                model=self.model, input=batch,
                extra_body={"input_type": input_type, "truncate": self.truncate},
            )
            return [d.embedding for d in resp.data]
        except Exception as e:  # noqa: BLE001
            raise EmbeddingsUnavailable(repr(e))


def _default_client(emb: dict):
    try:
        return http.openai_client(emb["base_url"], emb.get("api_key_env", "NVIDIA_API_KEY"))
    except ValueError as e:
        raise EmbeddingsUnavailable(str(e))

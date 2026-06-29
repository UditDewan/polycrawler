"""Hosted-API usage / quota monitor.

We never log raw call counts to a server, but every paid call leaves a cached
response on disk (llm_client / embeddings_client cache by input hash). Counting
cache entries is a free, honest proxy for "API calls made" and shows how much the
pre-filter + caching are saving against the free-tier quota.
"""
from __future__ import annotations

from pathlib import Path


def _count(d: str | Path) -> int:
    p = Path(d)
    return sum(1 for _ in p.glob("*.json")) if p.exists() else 0


def usage(cfg: dict) -> dict:
    llm_dir = cfg.get("llm", {}).get("cache_dir", "data/llm_cache")
    emb_dir = cfg.get("embeddings", {}).get("cache_dir", "data/emb_cache")
    llm_n, emb_n = _count(llm_dir), _count(emb_dir)
    return {
        "llm_calls_cached": llm_n,
        "embedding_calls_cached": emb_n,
        "total_api_calls": llm_n + emb_n,
        "note": "cached responses = paid calls made (each reused for free thereafter)",
    }

"""Local pre-filter — cut volume BEFORE any API call (quota is the bottleneck).

Drops: duplicates (content hash), too-short text, and off-topic text (no relevance
keyword). The keyword gate doubles as a cheap language filter — non-English football
chatter rarely contains 'injury'/'lineup'/'hamstring'.
# ponytail: keyword relevance gate, not a model. Swap in langdetect/a classifier
# only if non-English or off-topic noise gets through in practice.
Pure functions, unit-tested offline.
"""
from __future__ import annotations

import hashlib
import json
import re


def text_of(obs: dict) -> str:
    """Flatten an observation's JSON payload to plain text."""
    try:
        p = json.loads(obs.get("payload") or "{}")
    except (ValueError, TypeError):
        p = {}
    return " ".join(s for s in (p.get("title", ""), p.get("summary", ""), p.get("selftext", "")) if s).strip()


def content_hash(text: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", text.lower()).strip().encode("utf-8")).hexdigest()


def _has_keyword(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k.strip() in t for k in keywords if k.strip())


def passes(text: str, pf: dict) -> bool:
    if len(text) < pf.get("min_chars", 24):
        return False
    kws = pf.get("keywords") or []
    return _has_keyword(text, kws) if kws else True


def prefilter(observations: list[dict], pf: dict) -> tuple[list[dict], dict]:
    """Return (kept observations with a '_text' field, stats incl. drop_rate)."""
    kept: list[dict] = []
    seen: set[str] = set()
    n_dup = n_drop = 0
    for o in observations:
        text = text_of(o)
        h = content_hash(text)
        if h in seen:
            n_dup += 1
            continue
        seen.add(h)
        if not passes(text, pf):
            n_drop += 1
            continue
        kept.append({**o, "_text": text})
    total = len(observations)
    return kept, {
        "total": total, "dropped_dup": n_dup, "dropped_filter": n_drop, "kept": len(kept),
        "drop_rate": round(1 - len(kept) / total, 3) if total else 0.0,
    }

"""Point-in-time, recency- and credibility-weighted per-game retrieval.

The leakage guard lives in two places: a server-side Qdrant filter on `ts_ms`
(only signals before kickoff are candidates) AND a defensive assertion on the
results. A signal at/after kickoff must never enter a game's context bundle.
"""
from __future__ import annotations

from qdrant_client import models

from ..common.embeddings_client import EmbeddingsClient
from .index import COLLECTION, open_store


class RetrievalLeakageError(AssertionError):
    """A retrieved signal is at/after kickoff — point-in-time correctness was violated."""


def _credibility(source: str | None, weights: dict) -> float:
    s = (source or "").lower()
    for key, w in weights.items():
        if key != "default" and key in s:
            return w
    return weights.get("default", 0.5)


def assert_bundle_no_leakage(bundle: list[dict], kickoff_ms: int, *, strict: bool = True) -> None:
    bad = [c for c in bundle if (c["ts_ms"] >= kickoff_ms if strict else c["ts_ms"] > kickoff_ms)]
    if bad:
        raise RetrievalLeakageError(
            f"{len(bad)} retrieved signal(s) at/after kickoff (e.g. ts_ms={bad[0]['ts_ms']} "
            f"vs kickoff_ms={kickoff_ms})")


def retrieve_bundle(cfg, *, query_text, kickoff_ms, team=None, emb_client=None, qc=None,
                    top_k=None, strict=True) -> list[dict]:
    """Assemble a per-game context bundle as-of kickoff_ms. Only signals with
    ts < kickoff (strict) are eligible; results are re-ranked by
    similarity x recency-decay x source-credibility."""
    rc = cfg["retrieval"]
    top_k = top_k or rc.get("top_k", 8)
    emb = emb_client or EmbeddingsClient.from_config(cfg)
    qc = qc or open_store(cfg)

    qvec = emb.embed_one(query_text, input_type="query")
    rng = models.Range(lt=kickoff_ms) if strict else models.Range(lte=kickoff_ms)
    must = [models.FieldCondition(key="ts_ms", range=rng)]  # point-in-time filter
    if team:
        must.append(models.FieldCondition(key="team", match=models.MatchValue(value=team)))

    hits = qc.query_points(
        COLLECTION, query=qvec, query_filter=models.Filter(must=must),
        limit=rc.get("candidate_k", 50), with_payload=True,
    ).points

    half_ms = rc.get("recency_halflife_days", 3) * 86_400_000
    weights = rc.get("credibility", {})
    bundle = []
    for h in hits:
        p = h.payload
        age = max(kickoff_ms - p["ts_ms"], 0)
        recency = 0.5 ** (age / half_ms)
        cred = _credibility(p.get("source"), weights)
        bundle.append({
            "signal_id": p["signal_id"], "ts_ms": p["ts_ms"], "team": p.get("team"),
            "signal_type": p.get("signal_type"), "source": p.get("source"),
            "text": p.get("text"), "similarity": h.score,
            "score": h.score * recency * cred,
        })
    assert_bundle_no_leakage(bundle, kickoff_ms, strict=strict)  # belt-and-suspenders
    bundle.sort(key=lambda c: -c["score"])
    return bundle[:top_k]

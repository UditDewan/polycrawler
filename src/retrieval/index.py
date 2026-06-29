"""Embed extracted signals (their source text) as passages and index them into
Qdrant. Local in-process mode by default (no Docker). Payload carries `ts_ms` so
retrieval can enforce point-in-time correctness with a server-side filter.
"""
from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from ..common.embeddings_client import EmbeddingsClient
from ..extraction.prefilter import text_of

COLLECTION = "signals"


def open_store(cfg, *, location: str | None = None) -> QdrantClient:
    loc = location or cfg["vector_store"].get("path", "data/qdrant_local")
    return QdrantClient(location=":memory:") if loc == ":memory:" else QdrantClient(path=loc)


def ensure_collection(qc: QdrantClient, dim: int) -> None:
    if not qc.collection_exists(COLLECTION):
        qc.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )


def _rows(con, version: str) -> list[dict]:
    sql = """
        SELECT s.signal_id, s.obs_id, epoch_ms(s.ts) AS ts_ms, s.team, s.signal_type,
               s.is_relevant, COALESCE(s.confidence, 0.0) AS confidence, o.source, o.payload
        FROM signals s JOIN observations o USING (obs_id)
        WHERE s.version = ?
    """
    cols = ["signal_id", "obs_id", "ts_ms", "team", "signal_type",
            "is_relevant", "confidence", "source", "payload"]
    return [dict(zip(cols, r)) for r in con.execute(sql, [version]).fetchall()]


def run(con, cfg, *, emb_client=None, qc=None, relevant_only=True) -> dict:
    version = cfg["extraction"].get("signals_version", "v1")
    emb = emb_client or EmbeddingsClient.from_config(cfg)
    qc = qc or open_store(cfg)
    ensure_collection(qc, emb.dim)

    rows = _rows(con, version)
    if relevant_only:
        rows = [r for r in rows if r["is_relevant"]]
    texts = [text_of(r) for r in rows]
    keep = [(r, t) for r, t in zip(rows, texts) if t]
    if not keep:
        return {"indexed": 0}

    vectors = emb.embed([t for _, t in keep], input_type="passage")
    points = [
        models.PointStruct(
            id=str(uuid5(NAMESPACE_URL, r["signal_id"])),
            vector=vec,
            payload={
                "signal_id": r["signal_id"], "obs_id": r["obs_id"], "ts_ms": int(r["ts_ms"]),
                "team": r["team"], "signal_type": r["signal_type"],
                "confidence": r["confidence"], "source": r["source"], "text": t,
            },
        )
        for (r, t), vec in zip(keep, vectors)
    ]
    qc.upsert(COLLECTION, points=points)
    return {"indexed": len(points)}


def main() -> None:
    import argparse

    from ..common import store
    from ..common.config import load_config

    ap = argparse.ArgumentParser(description="Embed signals and index them into Qdrant.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    con = store.connect(cfg["storage"]["duckdb_path"])
    store.create_tables(con)
    qc = open_store(cfg)
    try:
        print(run(con, cfg, qc=qc))
    finally:
        qc.close()  # flush local store + avoid the __del__-at-shutdown warning


if __name__ == "__main__":
    main()

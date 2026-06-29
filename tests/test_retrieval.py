"""Phase 3: embeddings client caching + point-in-time retrieval (the leakage guard
at the retrieval layer). Offline: a fake embeddings client + Qdrant in-memory."""
import hashlib
from types import SimpleNamespace

import pytest

from src.common.embeddings_client import EmbeddingsClient
from src.common.store import connect, create_tables, upsert_observations, upsert_signals
from src.retrieval import index, retrieve

DIM = 16
T0 = 1_750_000_000_000  # arbitrary ms epoch used as the sample kickoff


def _vec(text: str):
    b = hashlib.sha256(text.encode()).digest()
    return [x / 255.0 for x in b[:DIM]]


class _FakeEmb:
    def __init__(self):
        self.calls = 0

    def create(self, model, input, **_):
        self.calls += len(input)
        return SimpleNamespace(data=[SimpleNamespace(embedding=_vec(t)) for t in input])


class _FakeEmbClient:
    def __init__(self):
        self.embeddings = _FakeEmb()


def _emb(tmp_path):
    return EmbeddingsClient(model="fake", dim=DIM, cache_dir=str(tmp_path), client=_FakeEmbClient())


def test_embeddings_cache_avoids_repeat_calls(tmp_path):
    e = _emb(tmp_path)
    e.embed(["alpha", "beta"], input_type="passage")
    e.embed(["alpha", "beta"], input_type="passage")  # cache hit
    assert e._client.embeddings.calls == 2  # only the first pair hit the API


def _seed(con):
    obs = [
        dict(obs_id="a", ts="2025-06-15T12:00:00Z", source="bbc", kind="news",
             payload='{"title":"Star striker an injury doubt"}'),
        dict(obs_id="b", ts="2025-06-15T13:00:00Z", source="theguardian", kind="news",
             payload='{"title":"Defender ruled out, lineup change"}'),
        dict(obs_id="c", ts="2025-06-15T16:00:00Z", source="bbc", kind="news",
             payload='{"title":"Full-time reaction and post-match quotes"}'),
    ]
    upsert_observations(con, obs)
    # ts straddle a kickoff at T0: a,b before; c after.
    sigs = [
        dict(signal_id="v1:a", obs_id="a", ts_ms=None, team="Egypt", signal_type="injury_news",
             is_relevant=True, confidence=0.9, version="v1"),
        dict(signal_id="v1:b", obs_id="b", ts_ms=None, team="Egypt", signal_type="lineup",
             is_relevant=True, confidence=0.8, version="v1"),
        dict(signal_id="v1:c", obs_id="c", ts_ms=None, team="Egypt", signal_type="other",
             is_relevant=True, confidence=0.7, version="v1"),
    ]
    # set explicit ts around T0 (store coerces datetimes; use ISO strings here)
    from datetime import datetime, timezone
    for s, off in zip(sigs, (-7200, -3600, +3600)):  # seconds rel. to T0
        s.pop("ts_ms")
        s["ts"] = datetime.fromtimestamp(T0 / 1000 + off, tz=timezone.utc)
    upsert_signals(con, sigs)


def test_retrieval_is_point_in_time(tmp_path):
    con = connect()
    create_tables(con)
    _seed(con)
    cfg = {
        "extraction": {"signals_version": "v1"},
        "vector_store": {"path": ":memory:"},
        "retrieval": {"top_k": 5, "candidate_k": 50, "recency_halflife_days": 3,
                      "credibility": {"default": 0.5, "bbc": 0.9, "theguardian": 0.85}},
    }
    emb = _emb(tmp_path)
    qc = index.open_store(cfg, location=":memory:")
    assert index.run(con, cfg, emb_client=emb, qc=qc)["indexed"] == 3

    bundle = retrieve.retrieve_bundle(cfg, query_text="injury news", kickoff_ms=T0,
                                      emb_client=emb, qc=qc)
    ids = {c["signal_id"] for c in bundle}
    assert ids == {"v1:a", "v1:b"}          # 'c' (post-kickoff) excluded
    assert all(c["ts_ms"] < T0 for c in bundle)


def test_assert_bundle_no_leakage_catches_future_signal():
    with pytest.raises(retrieve.RetrievalLeakageError):
        retrieve.assert_bundle_no_leakage([{"ts_ms": T0 + 1, "signal_id": "x"}], T0)

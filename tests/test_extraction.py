"""Phase 2: pre-filter, schema validation, LLM client (cache/retry/quota), and the
resumable orchestrator. All offline via an injected fake client — zero API calls."""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.common.llm_client import LLMClient, LLMUnavailable
from src.common.store import connect, create_tables, upsert_observations
from src.extraction import extract, prefilter
from src.extraction.schema import ExtractedSignal

PF = {"kinds": ["news", "social"], "min_chars": 10,
      "keywords": ["injur", "hamstring", "lineup", "ruled out", "doubt"]}


def _obs(obs_id, title, kind="news"):
    return dict(obs_id=obs_id, ts="2026-06-25T09:00:00Z", source="t", kind=kind,
                match_id=None, payload=json.dumps({"title": title}))


# ---- pre-filter ----
def test_prefilter_dedup_and_keyword_gate():
    obs = [_obs("1", "Salah hamstring injury doubt"),
           _obs("2", "Salah hamstring injury doubt"),   # duplicate text
           _obs("3", "What a great atmosphere lol"),     # no keyword
           _obs("4", "x")]                               # too short
    kept, stats = prefilter.prefilter(obs, PF)
    assert [k["obs_id"] for k in kept] == ["1"]
    assert (stats["dropped_dup"], stats["dropped_filter"], stats["kept"]) == (1, 2, 1)
    assert 0 < stats["drop_rate"] <= 1


# ---- schema ----
def test_schema_rejects_out_of_range_confidence():
    ExtractedSignal(is_relevant=True, signal_type="injury_news", confidence=0.9)
    with pytest.raises(ValidationError):
        ExtractedSignal(is_relevant=True, signal_type="injury_news", confidence=1.5)


# ---- fake OpenAI-compatible client ----
GOOD = ('{"is_relevant":true,"signal_type":"injury_news","team":"Liverpool",'
        '"player":"Salah","sentiment":"negative","confidence":0.8,"rationale":"hamstring"}')


class _FakeChat:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = 0

    def create(self, **_):
        self.calls += 1
        out = self._outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=out))])


class _FakeClient:
    model = "fake/model"

    def __init__(self, outputs):
        self.chat = SimpleNamespace(completions=_FakeChat(outputs))


def test_llm_client_caches_second_call(tmp_path):
    fc = _FakeClient([GOOD])
    c = LLMClient(model="fake/model", cache_dir=str(tmp_path), client=fc)
    a = c.complete_json("sys", "Salah hamstring", schema=ExtractedSignal)
    b = c.complete_json("sys", "Salah hamstring", schema=ExtractedSignal)
    assert a.player == b.player == "Salah"
    assert fc.chat.completions.calls == 1  # second served from cache


def test_llm_client_corrective_retry_on_bad_json(tmp_path):
    fc = _FakeClient(["not json at all", GOOD])
    c = LLMClient(model="fake/model", cache_dir=str(tmp_path), client=fc, max_retries=3)
    out = c.complete_json("sys", "Salah hamstring injury", schema=ExtractedSignal)
    assert out.signal_type.value == "injury_news"
    assert fc.chat.completions.calls == 2


def test_llm_client_raises_unavailable_after_retries(tmp_path):
    fc = _FakeClient([RuntimeError("boom")] * 3)
    c = LLMClient(model="fake/model", cache_dir=str(tmp_path), client=fc, max_retries=3)
    with pytest.raises(LLMUnavailable):
        c.complete_json("sys", "Salah injury", schema=ExtractedSignal)


# ---- orchestrator ----
def test_extract_run_writes_signals_and_is_resumable(tmp_path):
    con = connect()
    create_tables(con)
    upsert_observations(con, [_obs("1", "Salah hamstring injury doubt"),
                              _obs("2", "Defender ruled out, lineup change")])
    cfg = {"extraction": {"prefilter": PF, "signals_version": "v1"},
           "llm": {"extract_model": "fake/model", "temperature": 0.0, "cache_dir": str(tmp_path)}}
    r = extract.run(con, cfg, client=LLMClient(
        model="fake/model", cache_dir=str(tmp_path), client=_FakeClient([GOOD, GOOD])))
    assert r["extracted"] == 2 and r["json_validity"] == 1.0
    assert con.sql("SELECT count(*) FROM signals").fetchone()[0] == 2

    # resumable: both obs already extracted for v1 -> nothing eligible, no client calls
    r2 = extract.run(con, cfg, client=LLMClient(
        model="fake/model", cache_dir=str(tmp_path), client=_FakeClient([])))
    assert r2["extracted"] == 0 and r2["prefilter"]["total"] == 0

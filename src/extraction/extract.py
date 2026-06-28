"""Phase 2 orchestration: eligible observations -> pre-filter -> hosted LLM (strict
JSON, validated) -> signals table.

Resumable: skips observations already extracted for this version (so a quota stop
or crash loses no work). Quota-graceful: on LLMUnavailable it stops the batch and
returns what landed. Each signal row records model id + params + ts for reproducibility.

    uv run python -m src.extraction.extract            # run over ingested text
    uv run python -m src.extraction.extract --limit 50

NOTE: this hits the hosted API. Confirm schema + pre-filter + model strings before
large runs (free-tier quota). Tests/demo inject a fake client — zero API calls.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from ..common import store
from ..common.config import load_config
from ..common.llm_client import LLMClient, LLMUnavailable
from . import prefilter
from .schema import ExtractedSignal

_SYSTEM = (
    "You label football (soccer) social-media and news text for match-prediction signal.\n"
    "Decide whether the text carries match-relevant information (injuries, availability,\n"
    "lineups, suspensions, transfers) versus banter / opinion / noise.\n"
    "Return ONLY a JSON object matching this schema (no prose, no markdown):\n"
    "{schema}\n"
    'If it is banter or has no predictive value, set is_relevant=false and signal_type="banter".\n'
    "sentiment is from the perspective of the team's match chances; confidence is your 0..1 certainty."
)


def system_prompt() -> str:
    return _SYSTEM.replace("{schema}", json.dumps(ExtractedSignal.model_json_schema()))


def _eligible(con, kinds, version, limit):
    ph = ",".join("?" * len(kinds))
    sql = (
        f"SELECT o.obs_id, epoch_ms(o.ts) AS ts_ms, o.match_id, o.payload "  # epoch_ms avoids pytz
        f"FROM observations o WHERE o.kind IN ({ph}) "
        f"AND NOT EXISTS (SELECT 1 FROM signals s WHERE s.obs_id = o.obs_id AND s.version = ?) "
        f"ORDER BY o.ts DESC"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    cols = ["obs_id", "ts_ms", "match_id", "payload"]
    return [dict(zip(cols, r)) for r in con.execute(sql, list(kinds) + [version]).fetchall()]


def run(con, cfg, *, client=None, limit=None) -> dict:
    ex = cfg["extraction"]
    pf = ex["prefilter"]
    version = ex.get("signals_version", "v1")
    rows = _eligible(con, pf.get("kinds", ["social", "news"]), version, limit)
    kept, stats = prefilter.prefilter(rows, pf)

    client = client or LLMClient.from_config(cfg)
    system = system_prompt()
    model = getattr(client, "model", cfg["llm"]["extract_model"])
    params = json.dumps({"temperature": cfg["llm"].get("temperature", 0.0)})
    now = datetime.now(timezone.utc)

    n_ok = n_invalid = 0
    stopped: str | bool = False
    for o in kept:
        try:
            sig = client.complete_json(system, o["_text"], schema=ExtractedSignal)
        except LLMUnavailable as e:
            stopped = str(e)  # quota/transport — stop, resume later (work so far is persisted)
            break
        if sig is None:
            n_invalid += 1  # never produced valid JSON -> quarantine (no row written)
            continue
        ts = datetime.fromtimestamp(o["ts_ms"] / 1000, tz=timezone.utc)
        store.upsert_signals(con, [dict(
            signal_id=f"{version}:{o['obs_id']}", obs_id=o["obs_id"], ts=ts,
            match_id=o.get("match_id"), is_relevant=sig.is_relevant,
            signal_type=sig.signal_type.value, team=sig.team, player=sig.player,
            sentiment=sig.sentiment.value, confidence=sig.confidence, rationale=sig.rationale,
            model=model, model_params=params, version=version, extracted_at=now,
        )])
        n_ok += 1

    labeled = n_ok + n_invalid
    return {
        "prefilter": stats, "extracted": n_ok, "invalid": n_invalid,
        "json_validity": round(n_ok / labeled, 3) if labeled else None, "stopped": stopped,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract structured signals from ingested text.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    con = store.connect(cfg["storage"]["duckdb_path"])
    store.create_tables(con)
    print(run(con, cfg, limit=args.limit))


if __name__ == "__main__":
    main()

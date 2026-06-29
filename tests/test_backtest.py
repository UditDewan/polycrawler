"""Phase 5: the backtest must (1) never train on a match at/after a predicted match
(no lookahead) and (2) make no live API calls. Both are asserted here."""
import json
from datetime import datetime, timedelta, timezone

from eval.backtest import _blocks, walk_forward
from src.common.store import connect, create_tables, upsert_matches, upsert_observations


def test_blocks_are_leak_safe():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = [{"match_id": str(i), "kickoff": base + timedelta(days=i)} for i in range(20)]
    seen = 0
    for train, block, cutoff in _blocks(rows, step=3, min_train=5):
        seen += 1
        assert all(t["kickoff"] < cutoff for t in train)   # training strictly before block
        assert all(b["kickoff"] >= cutoff for b in block)
    assert seen > 0


def _seed(con, n=60):
    base = datetime(2023, 8, 1, 12, tzinfo=timezone.utc)
    teams = ["A", "B", "C", "D", "E", "F"]
    matches, obs = [], []
    for i in range(n):
        season = "2023-2024" if i < n // 2 else "2024-2025"
        home = teams[i % 6]
        away = teams[(i + 2) % 6] if teams[(i // 6 + 1) % 6] == teams[i % 6] else teams[(i // 6 + 1) % 6]
        res = ["H", "D", "A"][i % 3]
        hg, ag = {"H": (2, 0), "D": (1, 1), "A": (0, 2)}[res]
        k = base + timedelta(days=i)
        mid = f"m{i:03d}"
        matches.append(dict(match_id=mid, league="E0", season=season, home=home, away=away,
                            kickoff=k, home_goals=hg, away_goals=ag, status="final", result=res))
        obs.append(dict(obs_id="odds_" + mid, ts=k, source="football-data.co.uk", kind="odds",
                        match_id=mid, payload=json.dumps({"B365H": "2.0", "B365D": "3.4", "B365A": "3.8"})))
    upsert_matches(con, matches)
    upsert_observations(con, obs)


def _boom(*_a, **_k):
    raise AssertionError("backtest must not call a live API")


def test_walk_forward_no_lookahead_and_no_api(monkeypatch):
    import src.common.embeddings_client as emb
    import src.common.llm_client as llm
    monkeypatch.setattr(llm, "_default_client", _boom)      # poison the API clients
    monkeypatch.setattr(emb, "_default_client", _boom)

    con = connect()
    create_tables(con)
    _seed(con, 60)
    preds = walk_forward(con, step=6, min_train=18)

    assert len(preds) > 0
    assert all(p["cutoff"] <= p["kickoff"] for p in preds)            # no lookahead
    for p in preds:                                                  # valid simplex
        assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1.0) < 1e-6

"""Point-in-time feature builder for match-outcome prediction.

The leakage-critical part is `build_features`: it walks matches in kickoff order
and computes each team's form from ONLY its prior matches — a match's features are
finalized *before* its own result is folded into the running history. The market
feature is the de-vigged Bet365 closing line (known at kickoff).

`build_features` is a pure function over match dicts, so the leakage property is
unit-tested offline (tests/test_prediction.py).
"""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from math import nan

# Order matters: the model matrix is built from these names.
FEATURE_COLS = [
    "home_ppg", "home_gf", "home_ga", "home_n",
    "away_ppg", "away_gf", "away_ga", "away_n",
    "home_rest", "away_rest",
    "market_p_home", "market_p_draw", "market_p_away",
]
_TARGET = {"H": 0, "D": 1, "A": 2}


def market_probs(h, d, a) -> tuple[float, float, float]:
    """De-vig 1X2 decimal odds into probabilities that sum to 1 (removes overround)."""
    try:
        inv = [1.0 / float(h), 1.0 / float(d), 1.0 / float(a)]
    except (TypeError, ValueError, ZeroDivisionError):
        return (nan, nan, nan)
    s = sum(inv)
    return (inv[0] / s, inv[1] / s, inv[2] / s)


def _agg(history: deque) -> dict:
    n = len(history)
    if n == 0:
        return {"ppg": nan, "gf": nan, "ga": nan, "n": 0}
    return {"ppg": sum(x[0] for x in history) / n,
            "gf": sum(x[1] for x in history) / n,
            "ga": sum(x[2] for x in history) / n, "n": n}


def _rest_days(prev: datetime | None, kickoff: datetime) -> float:
    return nan if prev is None else (kickoff - prev).total_seconds() / 86400.0


def build_features(matches: list[dict], *, form_window: int = 5) -> list[dict]:
    """matches: dicts with match_id, kickoff (aware dt), season, home, away, result
    ('H'/'D'/'A' or None), home_goals, away_goals, b365h/b365d/b365a. Returns one
    feature row per match (target=None for unplayed)."""
    hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=form_window))
    last_played: dict[str, datetime] = {}
    rows = []
    for m in sorted(matches, key=lambda x: x["kickoff"]):
        h, a = m["home"], m["away"]
        hf, af = _agg(hist[h]), _agg(hist[a])
        mp = market_probs(m.get("b365h"), m.get("b365d"), m.get("b365a"))
        rows.append({
            "match_id": m["match_id"], "season": m["season"], "kickoff": m["kickoff"],
            "target": _TARGET.get(m.get("result")),
            "home_ppg": hf["ppg"], "home_gf": hf["gf"], "home_ga": hf["ga"], "home_n": hf["n"],
            "away_ppg": af["ppg"], "away_gf": af["gf"], "away_ga": af["ga"], "away_n": af["n"],
            "home_rest": _rest_days(last_played.get(h), m["kickoff"]),
            "away_rest": _rest_days(last_played.get(a), m["kickoff"]),
            "market_p_home": mp[0], "market_p_draw": mp[1], "market_p_away": mp[2],
        })
        # Fold THIS match into history only after its row is finalized (point-in-time).
        res, hg, ag = m.get("result"), m.get("home_goals"), m.get("away_goals")
        if res in _TARGET and hg is not None and ag is not None:
            hp, ap = {"H": (3, 0), "D": (1, 1), "A": (0, 3)}[res]
            hist[h].append((hp, hg, ag))
            hist[a].append((ap, ag, hg))
            last_played[h] = last_played[a] = m["kickoff"]
    return rows

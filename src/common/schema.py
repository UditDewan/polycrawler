"""Point-in-time schema.

Two record types, both carrying the timestamp that makes leak-free joins possible:

  Match       - a fixture. `kickoff` is the as-of cutoff. Result fields
                (home_goals/away_goals/result) are POST-match: never use them as
                features for predicting that same match.
  Observation - any timestamped fact (news, social, odds, stat). `ts` is when the
                info became KNOWN. A feature for a match may only use observations
                with ts < kickoff.

These dataclasses are the logical contract; physical DuckDB tables live in store.py
(TIMESTAMPTZ columns enforce tz-awareness at the storage layer).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Match:
    match_id: str
    league: str
    home: str
    away: str
    kickoff: datetime               # tz-aware UTC; the as-of cutoff
    season: str | None = None
    # --- post-match only; do not feed these in as pre-kickoff features ---
    home_goals: int | None = None
    away_goals: int | None = None
    status: str = "scheduled"       # scheduled | final
    result: str | None = None       # 'H' | 'D' | 'A'


@dataclass
class Observation:
    obs_id: str
    ts: datetime                    # tz-aware UTC; when the info became known
    source: str
    kind: str                       # news | social | odds | stat
    match_id: str | None = None
    credibility: float | None = None
    payload: str | None = None      # raw text / JSON; specialized per source later

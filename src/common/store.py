"""DuckDB store. TIMESTAMPTZ columns store instants in UTC, so as-of comparisons
are timezone-correct regardless of how the data arrived. Inserts funnel every
timestamp through to_utc() — a single choke point that guarantees tz-aware UTC."""
from __future__ import annotations

from pathlib import Path

import duckdb

from .timeutils import to_utc

MATCH_COLS = ["match_id", "league", "season", "home", "away", "kickoff",
              "home_goals", "away_goals", "status", "result"]
OBS_COLS = ["obs_id", "ts", "source", "kind", "match_id", "credibility", "payload"]
SIGNALS_COLS = ["signal_id", "obs_id", "ts", "match_id", "is_relevant", "signal_type",
                "team", "player", "sentiment", "confidence", "rationale",
                "model", "model_params", "version", "extracted_at"]
_TS_COLS = {"kickoff", "ts", "extracted_at"}

DDL = """
CREATE TABLE IF NOT EXISTS matches (
    match_id   VARCHAR PRIMARY KEY,
    league     VARCHAR,
    season     VARCHAR,
    home       VARCHAR,
    away       VARCHAR,
    kickoff    TIMESTAMPTZ NOT NULL,       -- as-of cutoff
    home_goals INTEGER,                    -- post-match only
    away_goals INTEGER,                    -- post-match only
    status     VARCHAR DEFAULT 'scheduled',
    result     VARCHAR                     -- 'H' | 'D' | 'A', post-match only
);
CREATE TABLE IF NOT EXISTS observations (
    obs_id      VARCHAR PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL,      -- when the info became known (point-in-time)
    source      VARCHAR,
    kind        VARCHAR,                   -- news | social | odds | stat
    match_id    VARCHAR,
    credibility DOUBLE,
    payload     VARCHAR
);
CREATE TABLE IF NOT EXISTS signals (
    signal_id    VARCHAR PRIMARY KEY,    -- "{version}:{obs_id}"
    obs_id       VARCHAR,
    ts           TIMESTAMPTZ NOT NULL,   -- inherited from the source observation (point-in-time)
    match_id     VARCHAR,
    is_relevant  BOOLEAN,
    signal_type  VARCHAR,                -- injury_news | rumor | lineup | transfer | suspension | banter | other
    team         VARCHAR,
    player       VARCHAR,
    sentiment    VARCHAR,                -- positive | negative | neutral
    confidence   DOUBLE,
    rationale    VARCHAR,
    model        VARCHAR,                -- hosted model id used (reproducibility)
    model_params VARCHAR,
    version      VARCHAR,                -- extraction schema/prompt version
    extracted_at TIMESTAMPTZ
);
"""


def connect(path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(path)
    con.execute("SET TimeZone='UTC'")
    return con


def create_tables(con: duckdb.DuckDBPyConnection) -> None:
    for stmt in filter(str.strip, DDL.split(";")):
        con.execute(stmt)


_VERBS = {"insert": "INSERT", "replace": "INSERT OR REPLACE", "ignore": "INSERT OR IGNORE"}


def _insert(con, table: str, cols: list[str], rows: list[dict], *, mode: str = "insert") -> int:
    if not rows:
        return 0
    data = []
    for r in rows:
        row = []
        for c in cols:
            v = r.get(c)
            if c in _TS_COLS and v is not None:
                v = to_utc(v)               # enforce tz-aware UTC at the boundary
            row.append(v)
        data.append(row)
    placeholders = ",".join("?" * len(cols))
    con.executemany(
        f"{_VERBS[mode]} INTO {table} ({','.join(cols)}) VALUES ({placeholders})", data)
    return len(rows)


def insert_matches(con, rows: list[dict]) -> None:
    _insert(con, "matches", MATCH_COLS, rows)


def insert_observations(con, rows: list[dict]) -> None:
    _insert(con, "observations", OBS_COLS, rows)


# Idempotent variants for re-runnable ingestion (rely on the PRIMARY KEYs):
# matches REPLACE so post-match results overwrite the scheduled row; observations
# IGNORE because a captured fact is immutable once seen.
def upsert_matches(con, rows: list[dict]) -> int:
    return _insert(con, "matches", MATCH_COLS, rows, mode="replace")


def upsert_observations(con, rows: list[dict]) -> int:
    return _insert(con, "observations", OBS_COLS, rows, mode="ignore")


def upsert_signals(con, rows: list[dict]) -> int:
    return _insert(con, "signals", SIGNALS_COLS, rows, mode="ignore")

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
_TS_COLS = {"kickoff", "ts"}

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


def _insert(con, table: str, cols: list[str], rows: list[dict]) -> None:
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
    con.executemany(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})", data)


def insert_matches(con, rows: list[dict]) -> None:
    _insert(con, "matches", MATCH_COLS, rows)


def insert_observations(con, rows: list[dict]) -> None:
    _insert(con, "observations", OBS_COLS, rows)

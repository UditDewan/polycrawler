"""Point-in-time (as-of) joins — the leakage guard everything depends on.

Invariant: a feature for a match may only use observations whose `ts` is strictly
before `kickoff`. We lean on DuckDB's native ASOF JOIN rather than hand-rolling a
merge, because it is correct on the boundary/tie cases that hand-rolled merges get
wrong. `strict=True` (the default) means strictly-before (ts < kickoff); the
exactly-at-kickoff row counts as leakage.
"""
from __future__ import annotations

import duckdb

Source = "str | duckdb.DuckDBPyRelation"


class LeakageError(AssertionError):
    """Raised when a set of rows contains an observation at/after kickoff."""


def _rel(con, source) -> duckdb.DuckDBPyRelation:
    return con.sql(f"SELECT * FROM {source}") if isinstance(source, str) else source


def point_in_time_obs(con, matches="matches", observations="observations", *, strict=True):
    """Every (match, observation) pair where the observation predates kickoff.

    This is the leak-safe candidate set that pre-match features get aggregated from.
    """
    op = "<" if strict else "<="
    return con.sql(f"""
        SELECT m.match_id, m.kickoff, o.* EXCLUDE (match_id)
        FROM {matches} m
        JOIN {observations} o
          ON o.match_id = m.match_id
         AND o.ts {op} m.kickoff
    """)


def asof_latest(con, matches="matches", observations="observations", *, strict=True):
    """The single latest observation per match before kickoff (latest-value features,
    e.g. the freshest odds snapshot we were allowed to see)."""
    op = ">" if strict else ">="          # m.kickoff > o.ts  <=>  o.ts < m.kickoff
    return con.sql(f"""
        SELECT m.*, o.* EXCLUDE (match_id)
        FROM {matches} m
        ASOF LEFT JOIN {observations} o
          ON m.match_id = o.match_id
         AND m.kickoff {op} o.ts
    """)


def assert_no_leakage(con, source, *, kickoff_col="kickoff", ts_col="ts", strict=True) -> None:
    """Raise LeakageError if any row has ts at/after kickoff. `source` is a table/view
    name or a DuckDB relation. Feature builders and the backtest call this defensively."""
    op = "<" if strict else "<="
    bad = _rel(con, source).filter(
        f"{ts_col} IS NOT NULL AND NOT ({ts_col} {op} {kickoff_col})"
    ).aggregate(  # cast to text: avoids pulling pytz in just to format the message
        f"count(*) AS n, min({ts_col})::VARCHAR AS first_bad, max({kickoff_col})::VARCHAR AS ref")
    n, first_bad, ref = bad.fetchone()
    if n:
        boundary = "strict <" if strict else "<="
        raise LeakageError(
            f"{n} observation(s) at/after kickoff (boundary {boundary}); "
            f"e.g. ts={first_bad} vs kickoff={ref}"
        )


def demo() -> None:
    """Self-check: build one match + observations straddling kickoff, prove the
    as-of helpers exclude the at/after rows and the assertion catches a leaky join."""
    from .store import connect, create_tables, insert_matches, insert_observations

    kickoff = "2026-06-27T18:00:00Z"
    con = connect()
    create_tables(con)
    insert_matches(con, [dict(match_id="m1", league="WC", home="ARG", away="BRA",
                              kickoff=kickoff)])
    insert_observations(con, [
        dict(obs_id="o1", ts="2026-06-27T09:00:00Z", source="news", kind="news", match_id="m1"),
        dict(obs_id="o2", ts="2026-06-27T17:59:00Z", source="news", kind="news", match_id="m1"),  # latest pre-kick
        dict(obs_id="at", ts=kickoff, source="news", kind="news", match_id="m1"),                  # exactly kickoff -> leak
        dict(obs_id="post", ts="2026-06-27T20:00:00Z", source="news", kind="news", match_id="m1"), # after -> leak
        dict(obs_id="tz", ts="2026-06-27T14:00:00-05:00", source="news", kind="news", match_id="m1"),  # =19:00Z -> leak
    ])

    pit_ids = sorted(r[0] for r in point_in_time_obs(con).project("obs_id").fetchall())
    assert pit_ids == ["o1", "o2"], pit_ids
    assert asof_latest(con).project("obs_id").fetchone()[0] == "o2"
    assert_no_leakage(con, point_in_time_obs(con))            # filtered set is clean

    leaky = con.sql("SELECT o.*, m.kickoff FROM observations o JOIN matches m USING (match_id)")
    try:
        assert_no_leakage(con, leaky)
        raise SystemExit("FAIL: leaky raw join was not caught")
    except LeakageError:
        pass

    print("asof demo OK: point-in-time filter + ASOF latest exclude at/after-kickoff rows; "
          "leaky join is caught")


if __name__ == "__main__":
    demo()

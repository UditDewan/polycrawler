"""Leakage suite: these fail if post-kickoff data can reach a feature set.
This is the #1 correctness contract of the whole project."""
import pytest

from src.common.asof import (
    LeakageError,
    asof_latest,
    assert_no_leakage,
    point_in_time_obs,
)
from src.common.store import connect, create_tables, insert_matches, insert_observations

KICKOFF = "2026-06-27T18:00:00Z"


def _fixture():
    con = connect()
    create_tables(con)
    insert_matches(con, [dict(match_id="m1", league="WC", home="ARG", away="BRA",
                              kickoff=KICKOFF)])
    insert_observations(con, [
        dict(obs_id="o1", ts="2026-06-27T09:00:00Z", source="news", kind="news", match_id="m1"),
        dict(obs_id="o2", ts="2026-06-27T17:59:00Z", source="news", kind="news", match_id="m1"),  # latest pre-kick
        dict(obs_id="at", ts=KICKOFF, source="news", kind="news", match_id="m1"),                  # exactly kickoff
        dict(obs_id="post", ts="2026-06-27T20:00:00Z", source="news", kind="news", match_id="m1"), # after
        # tz trap: 14:00 at -05:00 is 19:00Z -> AFTER kickoff despite the smaller wall-clock
        dict(obs_id="tz", ts="2026-06-27T14:00:00-05:00", source="news", kind="news", match_id="m1"),
    ])
    return con


def test_point_in_time_excludes_at_and_after():
    con = _fixture()
    ids = sorted(r[0] for r in point_in_time_obs(con).project("obs_id").fetchall())
    assert ids == ["o1", "o2"]            # 'at', 'post', 'tz' all dropped


def test_filtered_set_passes_leak_assert():
    con = _fixture()
    assert_no_leakage(con, point_in_time_obs(con))   # must not raise


def test_asof_latest_is_last_strictly_before_kickoff():
    con = _fixture()
    assert asof_latest(con).project("obs_id").fetchone()[0] == "o2"


def test_raw_join_leaks_and_is_caught():
    con = _fixture()
    leaky = con.sql("SELECT o.*, m.kickoff FROM observations o JOIN matches m USING (match_id)")
    with pytest.raises(LeakageError):
        assert_no_leakage(con, leaky)


def test_inclusive_boundary_keeps_kickoff_row_but_strict_assert_flags_it():
    con = _fixture()
    ids = sorted(r[0] for r in point_in_time_obs(con, strict=False).project("obs_id").fetchall())
    assert ids == ["at", "o1", "o2"]      # <= keeps the exactly-at-kickoff row
    with pytest.raises(LeakageError):     # ...which strict-mode then flags as leakage
        assert_no_leakage(con, point_in_time_obs(con, strict=False), strict=True)

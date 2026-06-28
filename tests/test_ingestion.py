"""Phase 1 ingestion: parsing + timestamp correctness + idempotency. All offline."""
import json
from types import SimpleNamespace

from src.common.store import connect, create_tables
from src.ingestion import football_data, news_rss, reddit

# Two fixtures straddling the BST/GMT boundary to prove DST-correct UTC conversion.
CSV = (
    "Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
    "17/08/2024,15:00,Liverpool,Brentford,2,0,H,1.50,4.50,6.00\n"   # BST (UTC+1) -> 14:00Z
    "26/12/2024,15:00,Arsenal,Ipswich,3,1,H,1.30,5.00,9.00\n"        # GMT (UTC+0) -> 15:00Z
)


def test_parse_csv_kickoff_is_dst_correct_utc():
    matches, obs = football_data.parse_csv(CSV, "E0", "2024-2025")
    by_home = {m["home"]: m for m in matches}
    summer, winter = by_home["Liverpool"]["kickoff"], by_home["Arsenal"]["kickoff"]
    assert summer.hour == 14 and str(summer.tzinfo) == "UTC"   # BST shifted by 1h
    assert winter.hour == 15                                   # GMT unchanged
    assert by_home["Liverpool"]["result"] == "H"
    assert by_home["Liverpool"]["match_id"] == "E0_2024-2025_2024-08-17_liverpool_brentford"
    # one odds observation per match, timestamped at kickoff
    assert len(obs) == 2 and obs[0]["kind"] == "odds" and obs[0]["ts"] == summer
    assert json.loads(obs[0]["payload"])["B365H"] == "1.50"


def test_football_data_upsert_is_idempotent():
    matches, obs = football_data.parse_csv(CSV, "E0", "2024-2025")
    con = connect()
    create_tables(con)
    for _ in range(2):  # run twice
        football_data_upsert(con, matches, obs)
    assert con.sql("SELECT count(*) FROM matches").fetchone()[0] == 2
    assert con.sql("SELECT count(*) FROM observations").fetchone()[0] == 2


def football_data_upsert(con, matches, obs):
    from src.common import store
    store.upsert_matches(con, matches)
    store.upsert_observations(con, obs)


RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>
<item><title>Salah injury doubt</title><link>https://ex.com/a</link>
<description>hamstring</description><pubDate>Wed, 25 Jun 2025 09:30:00 GMT</pubDate></item>
<item><title>Banter</title><link>https://ex.com/b</link>
<pubDate>Wed, 25 Jun 2025 10:00:00 +0000</pubDate></item></channel></rss>"""


def test_parse_feed_timestamps_and_idempotency():
    obs = news_rss.parse_feed(RSS, source="testfeed")
    assert len(obs) == 2
    first = next(o for o in obs if json.loads(o["payload"])["link"] == "https://ex.com/a")
    assert first["ts"].hour == 9 and first["ts"].minute == 30 and str(first["ts"].tzinfo) == "UTC"
    assert first["kind"] == "news"
    con = connect()
    create_tables(con)
    from src.common import store
    store.upsert_observations(con, obs)
    store.upsert_observations(con, obs)  # again
    assert con.sql("SELECT count(*) FROM observations").fetchone()[0] == 2


def test_reddit_submission_mapping():
    sub = SimpleNamespace(
        id="abc123", created_utc=1_700_000_000, title="Team X lineup leaked",
        selftext="details", permalink="/r/soccer/comments/abc123/x/",
    )
    o = reddit.submission_to_obs(sub, "soccer")
    assert o["obs_id"] == "reddit_abc123"
    assert o["kind"] == "social" and o["source"] == "reddit/r/soccer"
    assert str(o["ts"].tzinfo) == "UTC"
    assert json.loads(o["payload"])["url"].endswith("/r/soccer/comments/abc123/x/")

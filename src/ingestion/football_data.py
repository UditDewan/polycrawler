"""football-data.co.uk collector: historical match results + closing odds CSVs.

No auth, no key. URL pattern: /mmz4281/{seasoncode}/{league}.csv where seasoncode
is the two-digit start+end year (2024-2025 -> "2425"). Lands one `matches` row per
fixture and one `odds` observation (Bet365 closing 1X2) timestamped at kickoff.

Kickoff times in the CSV are UK local (Europe/London); we localize with zoneinfo
(DST-correct) and store UTC. parse_csv() is pure/offline so it is unit-tested.
"""
from __future__ import annotations

import csv
import io
import json
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

from ..common import http, store

BASE = "https://www.football-data.co.uk/mmz4281"
UK = ZoneInfo("Europe/London")


def season_code(season: str) -> str:
    start, end = season.split("-")
    return start[2:] + end[2:]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _parse_date(s: str) -> date:
    s = s.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date {s!r}")


def _kickoff(d: date, time_: str | None) -> datetime:
    if time_ and time_.strip():
        hh, mm = (time_.strip().split(":") + ["0"])[:2]
        naive = datetime(d.year, d.month, d.day, int(hh), int(mm))
    else:
        # ponytail: older rows lack a Time column -> noon UK placeholder. Precise
        # enough for results; refine if intraday signals ever attach to old games.
        naive = datetime(d.year, d.month, d.day, 12, 0)
    return naive.replace(tzinfo=UK).astimezone(ZoneInfo("UTC"))


def _int(v: str | None) -> int | None:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def parse_csv(text: str, league: str, season: str) -> tuple[list[dict], list[dict]]:
    matches: list[dict] = []
    obs: list[dict] = []
    for row in csv.DictReader(io.StringIO(text)):
        home, away, datestr = row.get("HomeTeam"), row.get("AwayTeam"), row.get("Date")
        if not (home and away and datestr):
            continue
        d = _parse_date(datestr)
        kickoff = _kickoff(d, row.get("Time"))
        match_id = f"{league}_{season}_{d.isoformat()}_{_slug(home)}_{_slug(away)}"
        ftr = (row.get("FTR") or "").strip() or None
        matches.append(dict(
            match_id=match_id, league=league, season=season, home=home, away=away,
            kickoff=kickoff, home_goals=_int(row.get("FTHG")), away_goals=_int(row.get("FTAG")),
            status="final" if ftr else "scheduled", result=ftr,
        ))
        odds = {k: row[k] for k in ("B365H", "B365D", "B365A") if row.get(k)}
        if odds:
            # Closing line == the kickoff snapshot; join with <= kickoff at feature time.
            obs.append(dict(
                obs_id=f"odds_{match_id}", ts=kickoff, source="football-data.co.uk",
                kind="odds", match_id=match_id, payload=json.dumps(odds),
            ))
    return matches, obs


def collect(con, cfg: dict) -> dict:
    hist = cfg["history"]
    n_matches = n_odds = 0
    for league in hist["leagues"]:
        for season in hist["seasons"]:
            url = f"{BASE}/{season_code(season)}/{league}.csv"
            text = http.get(url).text
            matches, obs = parse_csv(text, league, season)
            n_matches += store.upsert_matches(con, matches)
            n_odds += store.upsert_observations(con, obs)
    return {"matches": n_matches, "odds_obs": n_odds}

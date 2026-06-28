"""News RSS collector. Fetches feeds via the shared http getter (so feedparser
uses our OS-trust transport, not its own urllib) and lands one observation per
entry, timestamped at the entry's published time (UTC). parse_feed() is offline-testable."""
from __future__ import annotations

import hashlib
import json
from calendar import timegm
from datetime import datetime, timezone

import feedparser

from ..common import http, store


def _obs_id(link: str) -> str:
    return "news_" + hashlib.sha1(link.encode("utf-8")).hexdigest()[:16]


def _entry_ts(entry) -> datetime | None:
    t = entry.get("published_parsed") or entry.get("updated_parsed")
    if not t:
        return None
    # feedparser normalizes to UTC struct_time; timegm reads it as UTC.
    return datetime.fromtimestamp(timegm(t), tz=timezone.utc)


def parse_feed(content: bytes | str, source: str) -> list[dict]:
    feed = feedparser.parse(content)
    obs: list[dict] = []
    for e in feed.entries:
        link = e.get("link")
        ts = _entry_ts(e)
        if not link or ts is None:
            continue
        obs.append(dict(
            obs_id=_obs_id(link), ts=ts, source=source, kind="news", match_id=None,
            payload=json.dumps({"title": e.get("title", ""), "summary": e.get("summary", ""), "link": link}),
        ))
    return obs


def collect(con, cfg: dict) -> dict:
    feeds = cfg["sources"]["news_rss"].get("feeds", [])
    if not feeds:
        return {"skipped": "no feeds configured"}
    n = 0
    for url in feeds:
        n += store.upsert_observations(con, parse_feed(http.get(url).content, source=url))
    return {"news_obs": n}

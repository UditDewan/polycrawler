"""Reddit social collector via PRAW. Skips gracefully (no error) when creds are
absent or praw isn't installed, so the pipeline runs without Reddit configured.
submission_to_obs() is a pure mapping, unit-tested with a fake submission."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from ..common import store


def _creds() -> tuple[str | None, str | None, str]:
    return (
        os.environ.get("REDDIT_CLIENT_ID"),
        os.environ.get("REDDIT_CLIENT_SECRET"),
        os.environ.get("REDDIT_USER_AGENT", "polycrawler/0.1"),
    )


def submission_to_obs(sub, subreddit: str) -> dict:
    ts = datetime.fromtimestamp(sub.created_utc, tz=timezone.utc)  # created_utc is epoch UTC
    return dict(
        obs_id="reddit_" + sub.id, ts=ts, source=f"reddit/r/{subreddit}", kind="social",
        match_id=None,
        payload=json.dumps({
            "title": sub.title,
            "selftext": getattr(sub, "selftext", "") or "",
            "url": "https://reddit.com" + sub.permalink,
        }),
    )


def collect(con, cfg: dict) -> dict:
    client_id, secret, user_agent = _creds()
    if not (client_id and secret):
        return {"skipped": "no REDDIT_CLIENT_ID/SECRET in env"}
    try:
        import praw
    except ImportError:
        return {"skipped": "praw not installed"}

    reddit = praw.Reddit(client_id=client_id, client_secret=secret, user_agent=user_agent)
    rc = cfg["sources"]["reddit"]
    limit = rc.get("limit", 100)
    n = 0
    for sr in rc.get("subreddits", []):
        obs = [submission_to_obs(s, sr) for s in reddit.subreddit(sr).new(limit=limit)]
        n += store.upsert_observations(con, obs)
    return {"reddit_obs": n}

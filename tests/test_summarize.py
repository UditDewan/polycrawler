"""Scrape-and-summarize: reads recent news from the store and digests it. Offline
via a stub client (no API)."""
import json

from src import summarize
from src.common.store import connect, create_tables, upsert_observations


def _news(i, title, kind="news"):
    return dict(obs_id=f"n{i}", ts=f"2026-06-25T09:0{i}:00Z", source="feed", kind=kind,
                payload=json.dumps({"title": title}))


class _StubClient:
    def __init__(self):
        self.calls = 0
        self.last_user = None

    def complete_text(self, system, user, *, max_tokens=None):
        self.calls += 1
        self.last_user = user
        return "## Injuries & availability\n- Salah a doubt"


def test_recent_news_and_prompt():
    con = connect()
    create_tables(con)
    upsert_observations(con, [_news(1, "Salah injury doubt"), _news(2, "Arsenal sign striker")])
    items = summarize.recent_news(con, 10)
    assert set(items) == {"Salah injury doubt", "Arsenal sign striker"}
    prompt = summarize.build_prompt(items)
    assert prompt.startswith("Headlines:") and "Salah injury doubt" in prompt


def test_summarize_feeds_headlines_to_client():
    con = connect()
    create_tables(con)
    upsert_observations(con, [_news(1, "Salah injury doubt")])
    stub = _StubClient()
    digest, n = summarize.summarize(con, {"llm": {}}, client=stub)
    assert n == 1 and stub.calls == 1
    assert "Injuries" in digest and "Salah injury doubt" in stub.last_user


def test_summarize_empty_store_returns_none():
    con = connect()
    create_tables(con)
    digest, n = summarize.summarize(con, {"llm": {}}, client=_StubClient())
    assert digest is None and n == 0

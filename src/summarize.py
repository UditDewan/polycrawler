"""Scrape soccer news and summarize it — the project's primary purpose.

The ingestion collectors land timestamped news/social text in DuckDB; this turns the
most recent items into a concise, themed digest via the hosted LLM.

    uv run python -m src.summarize            # digest the news already in the store
    uv run python -m src.summarize --scrape   # scrape fresh news first, then digest
    uv run python -m src.summarize --limit 40 --out digest.md

Scraping needs no key; the summary step needs NVIDIA_API_KEY in .env.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .common import store
from .common.config import load_config
from .common.llm_client import LLMClient
from .extraction.prefilter import text_of

SYSTEM = (
    "You are a football (soccer) news editor. Given a list of recent headlines, write a "
    "concise markdown digest. Open with a one-sentence overview, then group items under "
    "short thematic headings (e.g. Injuries & availability, Transfers & rumours, Results & "
    "fixtures, Managers, Other) as tight one-line bullets. Be factual; use only what the "
    "input contains and omit anything unclear — do not invent."
)


def recent_news(con, limit: int, kinds=("news", "social")) -> list[str]:
    ph = ",".join("?" * len(kinds))
    rows = con.execute(
        f"SELECT payload FROM observations WHERE kind IN ({ph}) "
        f"ORDER BY ts DESC LIMIT {int(limit)}", list(kinds),
    ).fetchall()
    return [t for (payload,) in rows if (t := text_of({"payload": payload}))]


def build_prompt(items: list[str]) -> str:
    return "Headlines:\n" + "\n".join(f"- {t}" for t in items)


def summarize(con, cfg, *, limit: int = 30, client=None, max_tokens: int = 1200):
    """Return (markdown_digest, n_items). digest is None when there's nothing to summarize."""
    items = recent_news(con, limit)
    if not items:
        return None, 0
    client = client or LLMClient.from_config(cfg)
    return client.complete_text(SYSTEM, build_prompt(items), max_tokens=max_tokens), len(items)


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape soccer news and summarize it.")
    ap.add_argument("--scrape", action="store_true", help="scrape fresh news before summarizing")
    ap.add_argument("--limit", type=int, default=30, help="how many recent items to digest")
    ap.add_argument("--out", default=None, help="write the digest to this file instead of stdout")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    con = store.connect(cfg["storage"]["duckdb_path"])
    store.create_tables(con)
    if args.scrape:
        from .ingestion import news_rss
        print("scraping:", news_rss.collect(con, cfg))

    digest, n = summarize(con, cfg, limit=args.limit)
    if not digest:
        raise SystemExit("no news in the store — run with --scrape (or `python -m src.ingestion.run`) first")
    out = (f"# Soccer news digest\n_{n} items · "
           f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}_\n\n{digest}")
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"wrote {args.out} ({n} items)")
    else:
        print(out)


if __name__ == "__main__":
    main()

"""Ingestion runner. Runs the enabled collectors into the configured DuckDB store
and prints a summary. Idempotent: re-running upserts, so counts in the DB are stable.

    uv run python -m src.ingestion.run                 # all enabled sources
    uv run python -m src.ingestion.run --source news_rss --source football_data_couk
"""
from __future__ import annotations

import argparse

from ..common import store
from ..common.config import load_config
from . import football_data, news_rss, reddit

COLLECTORS = {
    "football_data_couk": football_data.collect,
    "news_rss": news_rss.collect,
    "reddit": reddit.collect,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run pluggable ingestion collectors.")
    ap.add_argument("--source", action="append", help="run only this source (repeatable)")
    ap.add_argument("--config", default=None, help="path to config YAML")
    args = ap.parse_args()

    cfg = load_config(args.config)
    con = store.connect(cfg["storage"]["duckdb_path"])
    store.create_tables(con)

    sources = cfg.get("sources", {})
    chosen = args.source or [
        name for name, sc in sources.items()
        if name in COLLECTORS and sc.get("enabled", True)
    ]

    print(f"Running: {', '.join(chosen) or '(none)'}")
    for name in chosen:
        fn = COLLECTORS.get(name)
        if fn is None:
            print(f"  {name}: unknown source")
            continue
        try:
            print(f"  {name}: {fn(con, cfg)}")
        except Exception as e:  # noqa: BLE001 - one bad source must not abort the rest
            print(f"  {name}: ERROR {e!r}")

    n_matches = con.sql("SELECT count(*) FROM matches").fetchone()[0]
    by_kind = con.sql("SELECT kind, count(*) FROM observations GROUP BY kind ORDER BY kind").fetchall()
    print(f"DB now: {n_matches} matches; observations by kind: {dict(by_kind)}")


if __name__ == "__main__":
    main()

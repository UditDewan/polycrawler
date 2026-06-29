"""Leak-free walk-forward backtest (the centerpiece).

Replays history in chronological blocks. For each block, the model is trained ONLY
on matches whose kickoff is strictly before the block's first kickoff — so no
prediction can ever see its own present or future (a conservative, leak-safe cut
that also avoids same-kickoff bleed). Reads only frozen DuckDB data: NO live API
calls. Two invariants are asserted here and tested in tests/test_backtest.py:
  1. every training match predates every predicted match (no lookahead)
  2. nothing in this path instantiates an LLM/embeddings client

    uv run python -m eval.backtest

Outputs Brier / log-loss / ECE over the whole replay, per-season calibration
(calibration-over-time), a reliability table, and persists per-match predictions
to `backtest_predictions` for Phase 6 paper trading.
"""
from __future__ import annotations

import numpy as np

from src.common import store
from src.common.config import load_config
from src.prediction.calibration import (
    brier_multiclass,
    ece,
    multiclass_log_loss,
    reliability_table,
)
from src.prediction.model import feature_rows, fit_calibrated, predict_calibrated


def _blocks(rows, step, min_train):
    """Yield (train, block, cutoff) where train = matches strictly before the block's
    first kickoff. rows must be sorted by kickoff."""
    n = len(rows)
    i = step
    while i < n:
        cutoff = rows[i]["kickoff"]
        train = [r for r in rows[:i] if r["kickoff"] < cutoff]
        if len(train) >= min_train:
            yield train, rows[i:i + step], cutoff
        i += step


def walk_forward(con, *, step: int = 38, min_train: int = 380, seed: int = 42) -> list[dict]:
    rows = sorted(feature_rows(con), key=lambda r: r["kickoff"])
    preds: list[dict] = []
    for train, block, cutoff in _blocks(rows, step, min_train):
        # leak guard: training strictly precedes the whole block
        assert max(r["kickoff"] for r in train) < cutoff <= min(b["kickoff"] for b in block), \
            "lookahead: a training match is not strictly before the block"
        cut = int(len(train) * 0.8)
        fit, calib = (train[:cut], train[cut:]) if train[cut:] else (train[:-1], train[-1:])
        clf, cal = fit_calibrated(fit, calib, seed=seed)
        P = predict_calibrated(clf, cal, block)
        for b, p in zip(block, P):
            preds.append(dict(
                match_id=b["match_id"], kickoff=b["kickoff"], season=b["season"],
                target=b["target"], cutoff=cutoff,
                p_home=float(p[0]), p_draw=float(p[1]), p_away=float(p[2]),
                m_home=b["market_p_home"], m_draw=b["market_p_draw"], m_away=b["market_p_away"],
            ))
    return preds


def _scores(P, y):
    return {"brier": brier_multiclass(P, y), "log_loss": multiclass_log_loss(P, y), "ece": ece(P, y)}


def report(preds: list[dict]) -> dict:
    P = np.array([[p["p_home"], p["p_draw"], p["p_away"]] for p in preds])
    M = np.array([[p["m_home"], p["m_draw"], p["m_away"]] for p in preds])
    y = np.array([p["target"] for p in preds])
    per_season = {}
    for s in sorted({p["season"] for p in preds}):
        idx = [k for k, p in enumerate(preds) if p["season"] == s]
        per_season[s] = {"n": len(idx), "model": _scores(P[idx], y[idx]),
                         "market": _scores(M[idx], y[idx])}
    return {"n": len(preds), "model": _scores(P, y), "market": _scores(M, y),
            "per_season": per_season, "reliability": reliability_table(P, y)}


PRED_DDL = """
CREATE TABLE IF NOT EXISTS backtest_predictions (
    match_id VARCHAR PRIMARY KEY, kickoff TIMESTAMPTZ, season VARCHAR, target INTEGER,
    p_home DOUBLE, p_draw DOUBLE, p_away DOUBLE, m_home DOUBLE, m_draw DOUBLE, m_away DOUBLE
)
"""


def persist(con, preds: list[dict]) -> None:
    con.execute(PRED_DDL)
    con.executemany(
        "INSERT OR REPLACE INTO backtest_predictions VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(p["match_id"], p["kickoff"], p["season"], p["target"], p["p_home"], p["p_draw"],
          p["p_away"], p["m_home"], p["m_draw"], p["m_away"]) for p in preds],
    )


def main() -> None:
    cfg = load_config()
    con = store.connect(cfg["storage"]["duckdb_path"])
    store.create_tables(con)
    preds = walk_forward(con)
    persist(con, preds)
    r = report(preds)
    print(f"Walk-forward backtest: {r['n']} predictions (leak-free, no API calls)\n")
    print(f"{'':14}{'Brier':>9}{'LogLoss':>9}{'ECE':>8}   (lower = better)")
    for name in ("model", "market"):
        s = r[name]
        print(f"{name:14}{s['brier']:9.4f}{s['log_loss']:9.4f}{s['ece']:8.4f}")
    print("\nCalibration over time (Brier / ECE):")
    print(f"{'season':14}{'n':>5}{'model_brier':>13}{'model_ece':>11}{'mkt_brier':>11}")
    for s, v in r["per_season"].items():
        print(f"{s:14}{v['n']:5d}{v['model']['brier']:13.4f}{v['model']['ece']:11.4f}"
              f"{v['market']['brier']:11.4f}")
    print("\nReliability (model, pooled one-vs-rest):")
    print(f"{'bin':>14}{'n':>7}{'pred':>8}{'observed':>10}")
    for lo, hi, n, pred, obs in r["reliability"]:
        print(f"  [{lo:.1f}, {hi:.1f}){n:7d}{pred:8.3f}{obs:10.3f}")


if __name__ == "__main__":
    main()

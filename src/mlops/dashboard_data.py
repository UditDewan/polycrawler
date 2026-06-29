"""Data prep for the dashboard — kept Streamlit-free so it's testable and the heavy
UI dep stays optional. Reads the frozen DuckDB tables the pipeline produced."""
from __future__ import annotations

import numpy as np

from ..prediction.calibration import ece, reliability_table


def _exists(con, table: str) -> bool:
    return con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()[0] > 0


def calibration_view(con) -> dict | None:
    """Reliability table + ECE from the persisted backtest predictions."""
    if not _exists(con, "backtest_predictions"):
        return None
    rows = con.execute(
        "SELECT p_home, p_draw, p_away, target FROM backtest_predictions"
    ).fetchall()
    if not rows:
        return None
    P = np.array([[r[0], r[1], r[2]] for r in rows])
    y = np.array([r[3] for r in rows], dtype=int)
    return {"n": len(rows), "ece": ece(P, y), "reliability": reliability_table(P, y)}


def pnl_view(con) -> dict | None:
    """Equity curve + summary from the paper-trading ledger."""
    if not _exists(con, "paper_trades"):
        return None
    rows = con.execute(
        "SELECT epoch_ms(kickoff), bankroll, pnl, stake, won FROM paper_trades ORDER BY kickoff"
    ).fetchall()
    if not rows:
        return None
    return {
        "n": len(rows),
        "equity": [(int(ms), float(bk)) for ms, bk, *_ in rows],
        "pnl": float(sum(r[2] for r in rows)),
        "staked": float(sum(r[3] for r in rows)),
        "hit_rate": sum(1 for r in rows if r[4]) / len(rows),
        "final_bankroll": float(rows[-1][1]),
    }

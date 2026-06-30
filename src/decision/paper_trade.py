"""(E) Paper-trading ledger. SIMULATION ONLY — no order placement, no real money.

Reads the leak-free backtest predictions, bets fractional-Kelly on positive-EV
outcomes priced at the *raw* (vig-inclusive) Bet365 closing odds, settles each bet
against the actual result, and compounds a paper bankroll in kickoff order.

Honest expectation: a model that is calibrated but BELOW the market should lose to
the vig — our "edges" are mostly noise against an efficient line. Positive P&L here
is a reason to suspect variance or leakage, not to celebrate.

    uv run python -m src.decision.paper_trade
"""
from __future__ import annotations

from datetime import datetime, timezone

from ..common import store
from ..common.config import load_config

OUTCOMES = ("H", "D", "A")   # index 0/1/2 == features._TARGET


def kelly_star(p: float, o: float) -> float:
    """Full-Kelly fraction of bankroll for decimal odds `o` at win prob `p`.
    Zero when the bet is not +EV (p*o <= 1)."""
    b = o - 1.0
    if b <= 0:
        return 0.0
    return max((p * o - 1.0) / b, 0.0)


def load_predictions(con) -> list[dict]:
    sql = """
        SELECT bp.match_id, epoch_ms(bp.kickoff) AS km, bp.target,
               bp.p_home, bp.p_draw, bp.p_away, o.payload
        FROM backtest_predictions bp
        JOIN observations o ON o.match_id = bp.match_id AND o.kind = 'odds'
    """
    import json
    out = []
    for mid, km, target, ph, pd, pa, payload in con.execute(sql).fetchall():
        od = json.loads(payload) if payload else {}
        try:
            odds = [float(od["B365H"]), float(od["B365D"]), float(od["B365A"])]
        except (KeyError, TypeError, ValueError):
            continue  # no usable price -> not tradeable
        out.append(dict(match_id=mid, kickoff=datetime.fromtimestamp(km / 1000, tz=timezone.utc),
                        target=target, probs=[ph, pd, pa], odds=odds))
    return out


def simulate(preds, *, bankroll=100.0, kelly_frac=0.25, min_edge=0.02) -> list[dict]:
    """One bet per match on its single best +EV outcome (ponytail: simplest defensible
    value-bet rule; upgrade to simultaneous-Kelly across outcomes only if needed)."""
    bankroll = float(bankroll)
    ledger = []
    for r in sorted(preds, key=lambda x: x["kickoff"]):
        best = None  # (edge, k)
        for k in range(3):
            o, p = r["odds"][k], r["probs"][k]
            edge = p * o - 1.0  # EV per unit staked at the real (vig-inclusive) price
            if edge > min_edge and (best is None or edge > best[0]):
                best = (edge, k)
        if best is None:
            continue
        edge, k = best
        f = min(kelly_frac * kelly_star(r["probs"][k], r["odds"][k]), 1.0)
        stake = min(f * bankroll, bankroll)
        if stake <= 0:
            continue
        won = r["target"] == k
        pnl = stake * (r["odds"][k] - 1.0) if won else -stake
        bankroll += pnl
        ledger.append(dict(
            match_id=r["match_id"], kickoff=r["kickoff"], outcome=OUTCOMES[k],
            odds=r["odds"][k], p_model=r["probs"][k], edge=edge, stake=stake,
            won=won, pnl=pnl, bankroll=bankroll,
        ))
    return ledger


def report(ledger, *, bankroll0=100.0) -> dict:
    n = len(ledger)
    if n == 0:
        return {"bets": 0, "final_bankroll": bankroll0}
    staked = sum(bet["stake"] for bet in ledger)
    pnl = sum(bet["pnl"] for bet in ledger)
    wins = sum(bet["won"] for bet in ledger)
    return {
        "bets": n, "staked": staked, "pnl": pnl,
        "roi": pnl / staked if staked else 0.0,
        "final_bankroll": ledger[-1]["bankroll"], "hit_rate": wins / n,
        "avg_model_prob_on_bets": sum(bet["p_model"] for bet in ledger) / n,  # vs hit_rate = calib check
        "avg_edge": sum(bet["edge"] for bet in ledger) / n,
    }


LEDGER_DDL = """
CREATE TABLE IF NOT EXISTS paper_trades (
    match_id VARCHAR PRIMARY KEY, kickoff TIMESTAMPTZ, outcome VARCHAR, odds DOUBLE,
    p_model DOUBLE, edge DOUBLE, stake DOUBLE, won BOOLEAN, pnl DOUBLE, bankroll DOUBLE
)
"""


def persist(con, ledger) -> None:
    con.execute("DROP TABLE IF EXISTS paper_trades")  # ledger is a full replay; rewrite wholesale
    con.execute(LEDGER_DDL)
    if ledger:
        con.executemany(
            "INSERT INTO paper_trades VALUES (?,?,?,?,?,?,?,?,?,?)",
            [(b["match_id"], b["kickoff"], b["outcome"], b["odds"], b["p_model"], b["edge"],
              b["stake"], b["won"], b["pnl"], b["bankroll"]) for b in ledger],
        )


def main() -> None:
    cfg = load_config()
    th = cfg.get("thresholds", {})
    bankroll0 = 100.0
    con = store.connect(cfg["storage"]["duckdb_path"])
    store.create_tables(con)
    preds = load_predictions(con)
    if not preds:
        raise SystemExit("no backtest_predictions found - run `python -m eval.backtest` first")
    ledger = simulate(preds, bankroll=bankroll0,
                      kelly_frac=th.get("kelly_fraction", 0.25), min_edge=th.get("min_edge", 0.02))
    persist(con, ledger)
    r = report(ledger, bankroll0=bankroll0)
    print(f"Paper trading over {len(preds)} backtest predictions (SIMULATED, no execution)\n")
    if r["bets"] == 0:
        print("No +EV bets cleared the threshold — nothing staked.")
        return
    print(f"  bets placed      : {r['bets']}")
    print(f"  total staked     : {r['staked']:.2f}")
    print(f"  P&L              : {r['pnl']:+.2f}   (start {bankroll0:.0f} -> {r['final_bankroll']:.2f})")
    print(f"  ROI              : {r['roi']:+.2%}")
    print(f"  hit rate         : {r['hit_rate']:.3f}")
    print(f"  avg model prob   : {r['avg_model_prob_on_bets']:.3f}   (vs hit rate {r['hit_rate']:.3f} = calibration of the bets)")
    print(f"  avg edge claimed : {r['avg_edge']:+.3f}")
    print("\nReminder: calibrated-but-below-market should lose to the vig. Positive P&L => suspect variance/leakage, not skill.")


if __name__ == "__main__":
    main()

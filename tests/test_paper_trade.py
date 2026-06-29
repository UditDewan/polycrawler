"""Phase 6: Kelly math + deterministic settlement/compounding. Pure functions, no DB."""
from datetime import datetime, timedelta, timezone

from src.decision.paper_trade import kelly_star, report, simulate

T0 = datetime(2024, 1, 1, 15, tzinfo=timezone.utc)


def test_kelly_star_known_values_and_no_negative():
    assert abs(kelly_star(0.6, 2.0) - 0.2) < 1e-12     # b=1 -> (1.2-1)/1
    assert abs(kelly_star(0.5, 3.0) - 0.25) < 1e-12    # b=2 -> (1.5-1)/2
    assert kelly_star(0.2, 2.0) == 0.0                 # -EV -> no bet
    assert kelly_star(0.9, 1.0) == 0.0                 # no payout (b<=0)


def _p(mid, days, target, probs, odds):
    return dict(match_id=mid, kickoff=T0 + timedelta(days=days), target=target,
                probs=probs, odds=odds)


def test_simulate_settles_and_compounds():
    preds = [
        _p("1", 0, 0, [0.6, 0.2, 0.2], [2.0, 4.0, 4.0]),   # home +EV (edge .2), HOME wins
        _p("2", 1, 1, [0.2, 0.2, 0.2], [2.0, 4.0, 4.0]),   # nothing +EV -> skip
        _p("3", 2, 2, [0.6, 0.2, 0.2], [2.0, 4.0, 4.0]),   # home +EV but AWAY wins -> loss
    ]
    ledger = simulate(preds, bankroll=100.0, kelly_frac=0.25, min_edge=0.02)
    assert [l["match_id"] for l in ledger] == ["1", "3"]    # match 2 skipped

    # match1: f=0.25*0.2=0.05, stake=5, win -> +5 -> bankroll 105
    assert abs(ledger[0]["stake"] - 5.0) < 1e-9
    assert ledger[0]["won"] and abs(ledger[0]["pnl"] - 5.0) < 1e-9
    assert abs(ledger[0]["bankroll"] - 105.0) < 1e-9
    # match3: stake off 105 -> 5.25, loss -> -5.25 -> bankroll 99.75
    assert not ledger[1]["won"]
    assert abs(ledger[1]["stake"] - 5.25) < 1e-9
    assert abs(ledger[1]["bankroll"] - 99.75) < 1e-9


def test_report_aggregates():
    preds = [_p("1", 0, 0, [0.6, 0.2, 0.2], [2.0, 4.0, 4.0])]
    r = report(simulate(preds), bankroll0=100.0)
    assert r["bets"] == 1 and r["hit_rate"] == 1.0
    assert abs(r["pnl"] - 5.0) < 1e-9 and abs(r["roi"] - 1.0) < 1e-9


def test_no_bets_when_market_efficient():
    # model == de-vigged fair probs, priced with vig -> never +EV beyond threshold
    preds = [_p("1", 0, 0, [0.50, 0.30, 0.20], [1.90, 3.10, 4.50])]
    assert simulate(preds, min_edge=0.02) == []

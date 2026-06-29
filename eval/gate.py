"""Calibration eval gate — the CI promotion gate.

Blocks promotion of any model whose calibration is WORSE than the committed
baseline. "Worse" = ECE, Brier, or log-loss regresses by more than `tol`.
Calibration (ECE) is the headline metric; Brier/log-loss are guards so a model
can't trade calibration for sharpness unnoticed.

    python -m eval.gate                       # compare current model to baseline; exit 1 if worse
    python -m eval.gate --update-baseline      # accept current metrics as the new baseline
    python -m eval.gate --metrics-file m.json  # compare a precomputed metrics file (CI, no data)
    python -m eval.gate --simulate-regression  # DEMO: feed a deliberately worse model -> exit 1

`check_calibration` is a pure function, tested both directions in tests/test_mlops.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASELINE = Path("eval/baseline_metrics.json")
GATED = ("ece", "brier", "log_loss")  # none may worsen beyond tol


def check_calibration(candidate: dict, baseline: dict, *, tol: float = 0.01) -> tuple[bool, list[str]]:
    """Return (ok, reasons). ok is False if any gated metric worsens by more than tol."""
    reasons = []
    for m in GATED:
        c, b = candidate.get(m), baseline.get(m)
        if c is None or b is None:
            reasons.append(f"{m} missing (candidate={c}, baseline={b})")
        elif c > b + tol:
            reasons.append(f"{m} worsened {b:.4f} -> {c:.4f} (> tol {tol})")
    return (len(reasons) == 0, reasons)


def current_metrics(con) -> dict:
    """Headline calibration metrics from a fresh leak-free walk-forward backtest."""
    from eval.backtest import report as backtest_report
    from eval.backtest import walk_forward

    r = backtest_report(walk_forward(con))
    return {"n": r["n"], **{k: r["model"][k] for k in GATED}}


def _emit(ok: bool, reasons: list[str], candidate: dict, baseline: dict) -> int:
    print(f"{'metric':10}{'baseline':>12}{'candidate':>12}")
    for m in GATED:
        print(f"{m:10}{baseline.get(m, float('nan')):12.4f}{candidate.get(m, float('nan')):12.4f}")
    if ok:
        print("\nGATE PASS — calibration did not regress; promotion allowed.")
    else:
        print("\nGATE FAIL — calibration regressed; promotion BLOCKED:")
        for r in reasons:
            print(f"  - {r}")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibration eval gate.")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--metrics-file", default=None, help="compare this metrics JSON (skips training)")
    ap.add_argument("--simulate-regression", action="store_true",
                    help="demo: a deliberately worse model must FAIL the gate (exit 1)")
    ap.add_argument("--tol", type=float, default=0.01)
    args = ap.parse_args()

    if args.simulate_regression:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        worse = {**baseline, "ece": baseline["ece"] + 3 * args.tol,
                 "brier": baseline["brier"] + 3 * args.tol}
        ok, reasons = check_calibration(worse, baseline, tol=args.tol)
        sys.exit(_emit(ok, reasons, worse, baseline))  # exits 1 — that's the point

    if args.metrics_file:
        candidate = json.loads(Path(args.metrics_file).read_text(encoding="utf-8"))
    else:
        from src.common import store
        from src.common.config import load_config
        con = store.connect(load_config()["storage"]["duckdb_path"])
        store.create_tables(con)
        candidate = current_metrics(con)

    if args.update_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
        print(f"baseline updated -> {BASELINE}: {candidate}")
        return

    if not BASELINE.exists():
        sys.exit("no baseline; run `python -m eval.gate --update-baseline` first")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    ok, reasons = check_calibration(candidate, baseline, tol=args.tol)
    sys.exit(_emit(ok, reasons, candidate, baseline))


if __name__ == "__main__":
    main()

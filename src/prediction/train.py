"""Train + calibrate the 3-way outcome model and report calibration vs the market.

Out-of-time split (no leakage): train on all but the latest season, validate on the
latest. Within train, a chronological tail is held out to fit the isotonic calibrator
(never the data the booster trained on). The market (de-vigged closing odds) is the
baseline we measure against — matching it is success; beating it should make us
suspect leakage.

    uv run python -m src.prediction.train
"""
from __future__ import annotations

import json
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

from ..common import store
from ..common.config import load_config
from .calibration import (
    OvRIsotonic,
    brier_multiclass,
    ece,
    multiclass_log_loss,
    reliability_table,
)
from .features import FEATURE_COLS, build_features

# LGBM + numpy interplay emits a cosmetic sklearn feature-name warning; silence it.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def load_matches(con) -> list[dict]:
    sql = """
        SELECT m.match_id, epoch_ms(m.kickoff) AS km, m.season, m.home, m.away, m.result,
               m.home_goals, m.away_goals, o.payload
        FROM matches m
        LEFT JOIN observations o ON o.match_id = m.match_id AND o.kind = 'odds'
    """
    out = []
    for mid, km, season, home, away, result, hg, ag, payload in con.execute(sql).fetchall():
        odds = json.loads(payload) if payload else {}
        out.append(dict(
            match_id=mid, kickoff=datetime.fromtimestamp(km / 1000, tz=timezone.utc),
            season=season, home=home, away=away, result=result, home_goals=hg, away_goals=ag,
            b365h=odds.get("B365H"), b365d=odds.get("B365D"), b365a=odds.get("B365A"),
        ))
    return out


def _matrix(rows):
    return np.array([[r[c] for c in FEATURE_COLS] for r in rows], dtype=float)


def _y(rows):
    return np.array([r["target"] for r in rows], dtype=int)


def run(con, *, seed: int = 42, save_to: str | None = "data/models/phase4.pkl") -> dict:
    rows = [r for r in build_features(load_matches(con))
            if r["target"] is not None and not np.isnan(r["market_p_home"])]
    seasons = sorted({r["season"] for r in rows})
    if len(seasons) < 2:
        raise SystemExit("need >=2 seasons of played matches with odds")
    test_season = seasons[-1]
    train = sorted((r for r in rows if r["season"] != test_season), key=lambda r: r["kickoff"])
    test = [r for r in rows if r["season"] == test_season]
    cut = int(len(train) * 0.8)
    fit, calib = train[:cut], train[cut:]  # chronological calibration holdout

    clf = LGBMClassifier(objective="multiclass", num_class=3, n_estimators=300,
                         learning_rate=0.03, num_leaves=31, random_state=seed, verbose=-1)
    clf.fit(_matrix(fit), _y(fit))
    cal = OvRIsotonic().fit(clf.predict_proba(_matrix(calib)), _y(calib))

    yt = _y(test)
    P_raw = clf.predict_proba(_matrix(test))
    P_cal = cal.transform(P_raw)
    P_mkt = np.array([[r["market_p_home"], r["market_p_draw"], r["market_p_away"]] for r in test])

    def scores(P):
        return {"brier": brier_multiclass(P, yt), "log_loss": multiclass_log_loss(P, yt),
                "ece": ece(P, yt)}

    report = {
        "test_season": test_season, "n_fit": len(fit), "n_calib": len(calib), "n_test": len(test),
        "model_raw": scores(P_raw), "model_calibrated": scores(P_cal), "market": scores(P_mkt),
        "mean_abs_edge_vs_market": float(np.abs(P_cal - P_mkt).mean()),
        "reliability_calibrated": reliability_table(P_cal, yt),
    }
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        with open(save_to, "wb") as f:
            pickle.dump({"clf": clf, "cal": cal, "features": FEATURE_COLS}, f)
    return report


def main() -> None:
    cfg = load_config()
    con = store.connect(cfg["storage"]["duckdb_path"])
    store.create_tables(con)
    r = run(con)
    print(f"Test season {r['test_season']}  |  n_test={r['n_test']}  "
          f"(fit={r['n_fit']}, calib={r['n_calib']})")
    print(f"\n{'model':18}{'Brier':>9}{'LogLoss':>9}{'ECE':>8}   (lower = better)")
    for name in ("model_raw", "model_calibrated", "market"):
        s = r[name]
        print(f"{name:18}{s['brier']:9.4f}{s['log_loss']:9.4f}{s['ece']:8.4f}")
    print(f"\nmean |edge| vs market: {r['mean_abs_edge_vs_market']:.4f}")
    print("\nReliability (calibrated model, pooled one-vs-rest):")
    print(f"{'bin':>14}{'n':>7}{'pred':>8}{'observed':>10}")
    for lo, hi, n, pred, obs in r["reliability_calibrated"]:
        print(f"  [{lo:.1f}, {hi:.1f}){n:7d}{pred:8.3f}{obs:10.3f}")


if __name__ == "__main__":
    main()

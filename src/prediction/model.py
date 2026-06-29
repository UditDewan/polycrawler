"""Shared data + model helpers for the predictor (used by train.py AND the backtest),
so the booster, its hyperparameters, and the feature matrix have one definition."""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import numpy as np
from lightgbm import LGBMClassifier

from .calibration import OvRIsotonic
from .features import FEATURE_COLS, build_features

# LGBM + numpy interplay emits a cosmetic sklearn feature-name warning; silence it.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

LGBM_PARAMS = dict(objective="multiclass", num_class=3, n_estimators=300,
                   learning_rate=0.03, num_leaves=31, verbose=-1)


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


def feature_rows(con) -> list[dict]:
    """Played matches with a market price, as point-in-time feature rows."""
    return [r for r in build_features(load_matches(con))
            if r["target"] is not None and not np.isnan(r["market_p_home"])]


def matrix(rows):
    return np.array([[r[c] for c in FEATURE_COLS] for r in rows], dtype=float)


def targets(rows):
    return np.array([r["target"] for r in rows], dtype=int)


def market_matrix(rows):
    return np.array([[r["market_p_home"], r["market_p_draw"], r["market_p_away"]] for r in rows],
                    dtype=float)


def fit_calibrated(fit_rows, calib_rows, *, seed: int = 42):
    clf = LGBMClassifier(random_state=seed, **LGBM_PARAMS)
    clf.fit(matrix(fit_rows), targets(fit_rows))
    cal = OvRIsotonic().fit(clf.predict_proba(matrix(calib_rows)), targets(calib_rows))
    return clf, cal


def predict_calibrated(clf, cal, rows):
    return cal.transform(clf.predict_proba(matrix(rows)))

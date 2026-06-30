"""Phase 4: point-in-time feature leakage, market de-vig, and calibration math.
Pure-function tests (no training, no DB)."""
import math
from datetime import datetime, timezone

import numpy as np

from src.prediction.calibration import OvRIsotonic, brier_multiclass, ece, multiclass_log_loss
from src.prediction.features import build_features, market_probs


def _m(mid, day, home, away, result, hg, ag, odds=(2.0, 3.5, 4.0)):
    return dict(match_id=mid, kickoff=datetime(2024, 1, day, 15, tzinfo=timezone.utc),
                season="2023-2024", home=home, away=away, result=result,
                home_goals=hg, away_goals=ag, b365h=odds[0], b365d=odds[1], b365a=odds[2])


def test_form_features_are_point_in_time():
    ms = [_m("1", 1, "A", "B", "H", 2, 0), _m("2", 8, "A", "C", "H", 1, 0),
          _m("3", 15, "B", "A", "A", 0, 1)]
    rows = {r["match_id"]: r for r in build_features(ms)}
    assert math.isnan(rows["1"]["home_ppg"])     # A had no prior games at m1
    assert rows["2"]["home_ppg"] == 3.0          # A won m1 -> 3 ppg going into m2
    assert rows["2"]["home_rest"] == 7.0         # 7 days since m1


def test_future_results_do_not_change_past_features():
    base = [_m("1", 1, "A", "B", "H", 2, 0), _m("2", 8, "A", "C", "H", 1, 0)]
    early = {r["match_id"]: r for r in build_features(base + [_m("3", 15, "B", "A", "A", 0, 1)])}
    flipped = {r["match_id"]: r for r in build_features(base + [_m("3", 15, "B", "A", "H", 5, 0)])}
    assert early["1"] == flipped["1"]            # changing m3 must not touch m1/m2
    assert early["2"] == flipped["2"]


def test_market_probs_devig_sums_to_one():
    p = market_probs(1.5, 4.5, 6.0)
    assert abs(sum(p) - 1.0) < 1e-9 and p[0] > p[2]   # short price -> higher prob
    assert all(math.isnan(x) for x in market_probs(None, 3.0, 4.0))


def test_calibration_outputs_simplex_and_brier_zero_is_perfect():
    P = np.array([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1]])
    y = np.array([2, 0, 1])
    Q = OvRIsotonic().fit(P, y).transform(P)
    assert np.allclose(Q.sum(axis=1), 1.0)
    perfect = np.eye(3)[y]
    assert brier_multiclass(perfect, y) == 0.0
    assert multiclass_log_loss(perfect, y) < 1e-9
    assert 0.0 <= ece(P, y) <= 1.0


def test_calibration_never_returns_a_zero_sum_row():
    # Regression: a low-confidence row used to map every class to 0 (sum 0, not a
    # distribution). Calibrated rows must always sum to 1.
    P = np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.34, 0.33, 0.33]])
    y = np.array([2, 1, 0, 2])
    cal = OvRIsotonic().fit(P, y)
    Q = cal.transform(np.array([[0.05, 0.05, 0.05]]))   # clips below every class's min
    assert abs(Q.sum() - 1.0) < 1e-9                    # was 0.0 before the fix

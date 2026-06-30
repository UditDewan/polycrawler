"""Phase 7: the eval gate (both directions), PSI drift, quota, dashboard data.
The gate test IS the 'CI fails on a deliberately worse model' deliverable."""
import numpy as np

from eval.gate import check_calibration
from src.common.store import connect, create_tables
from src.mlops import quota
from src.mlops.dashboard_data import calibration_view, pnl_view
from src.mlops.drift import feature_drift, psi
from src.prediction.features import FEATURE_COLS

BASE = {"ece": 0.040, "brier": 0.560, "log_loss": 0.950}


def test_gate_passes_on_equal_or_better():
    ok, reasons = check_calibration({"ece": 0.035, "brier": 0.555, "log_loss": 0.95}, BASE)
    assert ok and reasons == []


def test_gate_blocks_a_deliberately_worse_model():
    # ECE regresses well beyond tolerance -> promotion must be BLOCKED.
    ok, reasons = check_calibration({"ece": 0.090, "brier": 0.560, "log_loss": 0.95}, BASE)
    assert not ok and any("ece" in r for r in reasons)


def test_gate_tolerance_absorbs_noise():
    ok, _ = check_calibration({"ece": 0.045, "brier": 0.560, "log_loss": 0.95}, BASE, tol=0.01)
    assert ok  # +0.005 within tol


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    assert psi(ref, ref.copy()) < 0.01
    assert psi(ref, rng.normal(3, 1, 2000)) > 0.25   # big mean shift -> clear drift


def _row(v):
    return {c: v for c in FEATURE_COLS}


def test_feature_drift_flags_shifted_feature():
    rng = np.random.default_rng(1)
    ref = [{c: float(rng.normal()) for c in FEATURE_COLS} for _ in range(500)]
    cur = [dict(r) for r in ref]
    for r in cur:
        r["home_ppg"] += 5.0                          # shift one feature hard
    out = feature_drift(ref, cur)
    assert "home_ppg" in out["drifted"]
    assert out["max_psi"] == out["psi"]["home_ppg"]


def test_quota_counts_cache(tmp_path):
    (tmp_path / "a.json").write_text("[]")
    (tmp_path / "b.json").write_text("[]")
    u = quota.usage({"llm": {"cache_dir": str(tmp_path)}, "embeddings": {"cache_dir": "nope"}})
    assert u["llm_calls_cached"] == 2 and u["total_api_calls"] == 2


def test_dashboard_data_handles_empty_db():
    con = connect()
    create_tables(con)
    assert calibration_view(con) is None      # no backtest_predictions table yet
    assert pnl_view(con) is None

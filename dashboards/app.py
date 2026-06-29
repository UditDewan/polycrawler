"""Streamlit calibration + P&L dashboard (Phase 7 live deliverable).

    uv sync --group mlops
    uv run streamlit run dashboards/app.py

Renders the reliability diagram, paper-trading equity curve, feature drift, and
hosted-API usage from the frozen DuckDB tables. All data prep lives in
src/mlops/dashboard_data.py (Streamlit-free, tested); this file only renders.
"""
from __future__ import annotations

import streamlit as st

from src.common import store
from src.common.config import load_config
from src.mlops import quota
from src.mlops.dashboard_data import calibration_view, pnl_view

st.set_page_config(page_title="Polycrawler", layout="wide")
st.title("Polycrawler — calibration-first soccer forecaster")
st.caption("Success = calibration (Brier / log-loss / reliability), not profit. Paper-only.")

cfg = load_config()
con = store.connect(cfg["storage"]["duckdb_path"])
store.create_tables(con)

cal = calibration_view(con)
pnl = pnl_view(con)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Calibration (backtest)")
    if cal:
        st.metric("Pooled ECE", f"{cal['ece']:.4f}", help="lower = better calibrated")
        st.caption(f"{cal['n']} leak-free predictions")
        chart = {"predicted": [], "observed": []}
        for lo, hi, n, pred, obs in cal["reliability"]:
            chart["predicted"].append(pred)
            chart["observed"].append(obs)
        st.line_chart(chart, x="predicted", y="observed")
        st.caption("On the diagonal = perfectly calibrated.")
    else:
        st.info("No backtest yet — run `python -m eval.backtest`.")

with c2:
    st.subheader("Paper trading (simulated)")
    if pnl:
        st.metric("Final bankroll", f"{pnl['final_bankroll']:.2f}",
                  delta=f"{pnl['pnl']:+.2f}")
        st.caption(f"{pnl['n']} bets · hit rate {pnl['hit_rate']:.3f} · staked {pnl['staked']:.0f}")
        st.line_chart({"bankroll": [bk for _, bk in pnl["equity"]]})
        st.caption("Calibrated-but-below-market loses to the vig — by design.")
    else:
        st.info("No ledger yet — run `python -m src.decision.paper_trade`.")

st.subheader("Hosted-API usage")
st.json(quota.usage(cfg))

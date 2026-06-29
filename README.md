# Polycrawler

A **calibrated** soccer-outcome forecaster combining RAG + hosted-LLM signal
extraction (prompted, no training) + MLOps. Success = *when it says 70%, the event happens ~70% of the
time* (Brier score, log loss, reliability diagrams on a leak-free backtest).
Beating the market is upside, never the premise. **Paper-trading only — no live
execution, ever.**

## Scope (locked in Phase 0)

| | Choice | Why |
|---|---|---|
| Sport | Soccer | — |
| **History** (fit + leak-free backtest) | Club football, default EPL via **football-data.co.uk** | Many resolved games + historical closing odds → calibration is *provable* |
| **Live** (forward paper-trade) | **2026 World Cup**, prices from **Polymarket** | Fresh, liquid, exciting; inherently leak-free (match hasn't happened) |
| Storage / vectors | **DuckDB** + **Qdrant** | DuckDB's `ASOF JOIN` makes point-in-time correctness native |

> The World Cup alone can't prove calibration (~104 live matches, no historical
> replay). So we *fit and backtest on history* and *forward-test on the World Cup*
> with the same pipeline.

## The one invariant

Every feature for a match may use only observations with `ts < kickoff`. Enforced
architecturally: `TIMESTAMPTZ` columns, DuckDB `ASOF JOIN`, and an
`assert_no_leakage()` guard with a dedicated test suite (`tests/test_asof.py`).

## Run it

```bash
uv sync                              # create venv + install (duckdb, pyyaml, pytest)
# behind a TLS-intercepting proxy? add --system-certs to uv commands
uv run python -m src.common.asof     # as-of self-check (prints "asof demo OK ...")
uv run pytest -q                     # full suite incl. leakage + ingestion tests

# Phase 1 — ingestion (idempotent; safe to re-run):
uv run python -m src.ingestion.run                    # all enabled sources
uv run python -m src.ingestion.run --source news_rss  # just one source
#   football-data + news RSS need no auth. Reddit skips unless you copy
#   .env.example -> .env and set REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET.

# Phase 2 — signal extraction (hosted NVIDIA LLM; needs NVIDIA_API_KEY in .env):
uv run python -m src.extraction.extract --limit 20    # small run; scale up after sign-off

# Phase 3 — embed signals into Qdrant (local mode, no Docker); needs NVIDIA_API_KEY:
uv run python -m src.retrieval.index                  # index relevant signals as vectors

# Phase 4 — train + calibrate the outcome model (CPU, no API; reads EPL matches):
uv run python -m src.prediction.train                 # prints the calibration report

# Phase 5 — leak-free walk-forward backtest (no API; persists predictions):
uv run python -m eval.backtest                        # Brier/log-loss/reliability over history

# Phase 6 — paper trading over the backtest (SIMULATED, no execution):
uv run python -m src.decision.paper_trade             # fractional-Kelly ledger + P&L

# Phase 7 — MLOps: calibration eval gate, tracking, drift, dashboard
uv run python -m eval.gate                            # PASS/FAIL vs eval/baseline_metrics.json
uv run python -m eval.gate --simulate-regression      # DEMO: worse model -> GATE FAIL, exit 1
uv run python -m eval.gate --update-baseline          # accept current metrics as baseline
uv sync --group mlops                                 # adds mlflow + streamlit (optional, heavy)
uv run streamlit run dashboards/app.py                # live calibration + P&L dashboard

docker compose up -d qdrant          # optional Qdrant *server* (local mode used by default)
```

**What to look for:** the demo prints `asof demo OK ...`; `pytest` is green. The
leakage tests prove that an observation exactly at or after kickoff — including a
timezone trap (`14:00 -05:00` = `19:00Z`) — is excluded from features and that a
naive raw join is *caught*, not silently allowed.

## Layout

```
src/common/      # schema, time/as-of utilities, config  <- Phase 0 lives here
src/ingestion/   # (A) collectors            Phase 1
src/extraction/  # (B) hosted-LLM extractor  Phase 2
src/retrieval/   # (C) RAG                    Phase 3
src/prediction/  # (D) model + calibration   Phase 4
eval/            # leak-free backtest         Phase 5  (centerpiece)
src/decision/    # (E) paper trading          Phase 6
src/mlops/       # (F) tracking/drift/CI gate Phase 7
config/default.yaml
tests/           # incl. the leakage suite
```

## Status

- [x] **Phase 0** — scaffold + point-in-time schema/as-of utilities + leakage tests
- [x] **Phase 1** — pluggable ingestion (football-data.co.uk, news RSS, Reddit) → DuckDB, idempotent
- [x] **Phase 2** — signal extraction live (NVIDIA llama-3.3-70b → strict JSON; 47 signals, 100% valid, 68% pre-filter drop)
- [x] **Phase 3** — RAG retrieval (NVIDIA nv-embedqa-e5-v5 + Qdrant local; point-in-time, recency/credibility re-rank, leakage-guarded)
- [x] **Phase 4** — prediction + calibration: LightGBM + isotonic, out-of-time split; calibrated ECE 0.047 (raw 0.149), honestly below the market baseline
- [x] **Phase 5** — leak-free walk-forward backtest: 1,140 predictions, calibration stable over time (ECE ~0.04–0.05/season), honestly below market; no-lookahead + no-API tests
- [x] **Phase 6** — paper trading (fractional-Kelly, simulated ledger): −10% ROI / near-total ruin on 1,093 bets — the model loses to the vig, exactly as a calibrated-but-not-market-beating forecaster should
- [x] **Phase 7** — MLOps wrap: calibration eval gate (PASS on baseline / FAIL+exit-1 on a worse model) + GitHub Actions CI, MLflow tracking+registry, PSI feature drift, hosted-API quota monitor, Streamlit dashboard, DVC pipeline

**All 7 phases complete.** 41 tests pass (incl. the leakage suite + the gate's
block-a-worse-model test). The result is honest: the forecaster is *calibrated*
(ECE ~0.04, stable across seasons) and sits *below* an efficient market — proving
calibration, not beating the line, exactly as the brief framed success.

### What's real vs. toy
- **Real:** leak-free point-in-time architecture + dedicated leakage tests; live NVIDIA extraction & embeddings; walk-forward backtest; isotonic calibration; CI eval gate.
- **Honest limits:** RAG signals are live-World-Cup only (no contemporaneous text for historical EPL, so they're zero on the backtest); drift/quota are lightweight (PSI, cache-count) with Evidently notable as a swap-in; single league.

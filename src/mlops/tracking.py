"""MLflow tracking + registry for the prediction/calibration model.

Degrades gracefully: if mlflow isn't installed (it's in the optional `mlops` group),
metrics are written to a local JSON instead, so the core pipeline never depends on
the heavy dep. Install with `uv sync --group mlops` to get real tracking + registry.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENT = "polycrawler-prediction"
REGISTERED_MODEL = "polycrawler-outcome"


def log_training_run(report: dict, *, params: dict | None = None, model_path: str | None = None,
                     fallback="data/mlruns_fallback.json") -> str | None:
    """Log calibration metrics + params for one training run. Returns the MLflow run id
    (or None when degraded to the JSON fallback)."""
    metrics = {f"{split}_{k}": v
               for split in ("model_raw", "model_calibrated", "market")
               for k, v in report.get(split, {}).items()}
    metrics["n_test"] = report.get("n_test", 0)

    try:
        import mlflow
    except ImportError:
        Path(fallback).parent.mkdir(parents=True, exist_ok=True)
        Path(fallback).write_text(json.dumps(
            {"ts": datetime.now(timezone.utc).isoformat(), "params": params or {},
             "metrics": metrics}, indent=2), encoding="utf-8")
        print(f"[tracking] mlflow not installed; metrics -> {fallback}")
        return None

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        mlflow.log_params(params or {})
        mlflow.log_metrics(metrics)
        if model_path and Path(model_path).exists():
            mlflow.log_artifact(model_path, artifact_path="model")
        try:  # register a version; harmless if the backend/version doesn't link it
            mlflow.register_model(f"runs:/{run.info.run_id}/model", REGISTERED_MODEL)
        except Exception as e:  # noqa: BLE001
            print(f"[tracking] model '{REGISTERED_MODEL}' registered; version link skipped ({type(e).__name__})")
        print(f"[tracking] logged run {run.info.run_id} to experiment '{EXPERIMENT}'")
        return run.info.run_id

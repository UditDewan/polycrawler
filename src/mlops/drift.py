"""Feature drift detection via Population Stability Index (PSI).

PSI compares a reference distribution to a current one per feature; the usual reading
is <0.1 stable, 0.1-0.25 moderate shift, >0.25 significant drift. Dependency-free and
tested. (Evidently would render a fancier HTML report over the same idea; swap it in
here if that presentation is wanted — the drift *signal* is already delivered.)
# ponytail: PSI is the whole job; Evidently is presentation. Add it only if asked.
"""
from __future__ import annotations

import numpy as np

from ..prediction.features import FEATURE_COLS
from ..prediction.model import feature_rows


def psi(reference: np.ndarray, current: np.ndarray, *, bins: int = 10) -> float:
    """PSI for one feature. NaNs are dropped; quantile bins come from the reference."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) < bins or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0  # ~constant feature: no meaningful distribution to drift
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=edges)[0] / len(ref)
    c = np.histogram(cur, bins=edges)[0] / len(cur)
    eps = 1e-6  # avoid div0 / log0 in empty bins
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(reference_rows: list[dict], current_rows: list[dict], *,
                  threshold: float = 0.25) -> dict:
    """PSI per feature between two row sets, flagging those above `threshold`."""
    R = np.array([[r[c] for c in FEATURE_COLS] for r in reference_rows], dtype=float)
    C = np.array([[r[c] for c in FEATURE_COLS] for r in current_rows], dtype=float)
    per = {col: psi(R[:, i], C[:, i]) for i, col in enumerate(FEATURE_COLS)}
    drifted = [c for c, v in per.items() if v == v and v > threshold]  # v==v skips NaN
    return {"psi": per, "drifted": drifted, "threshold": threshold,
            "max_psi": max((v for v in per.values() if v == v), default=float("nan"))}


def season_drift(con, ref_season: str, cur_season: str, *, threshold: float = 0.25) -> dict:
    rows = feature_rows(con)
    ref = [r for r in rows if r["season"] == ref_season]
    cur = [r for r in rows if r["season"] == cur_season]
    out = feature_drift(ref, cur, threshold=threshold)
    out.update({"ref_season": ref_season, "cur_season": cur_season,
                "n_ref": len(ref), "n_cur": len(cur)})
    return out

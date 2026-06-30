"""Calibration (one-vs-rest isotonic) + proper scoring for 3-way outcomes.

These are the calibration-first scoreboard: Brier, log loss, ECE, and a pooled
reliability table. Lower Brier/log-loss/ECE = better calibrated.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

CLASSES = ("H", "D", "A")


class OvRIsotonic:
    """Fit one isotonic regressor per class on held-out predictions, then renormalize
    so calibrated probabilities still sum to 1."""

    def __init__(self):
        self.iso: list[IsotonicRegression] = []

    def fit(self, P: np.ndarray, y: np.ndarray) -> "OvRIsotonic":
        self.iso = []
        for k in range(P.shape[1]):
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(P[:, k], (y == k).astype(float))
            self.iso.append(ir)
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        Q = np.column_stack([self.iso[k].transform(P[:, k]) for k in range(P.shape[1])])
        s = Q.sum(axis=1, keepdims=True)
        # A low-confidence row can have every class's isotonic map to 0 -> the row
        # sums to 0, which is NOT a probability distribution and would silently
        # corrupt Brier/log-loss/ECE. Fall back to the raw (already-normalized) probs.
        degenerate = (s[:, 0] == 0)
        if degenerate.any():
            Q[degenerate] = P[degenerate]
            s = Q.sum(axis=1, keepdims=True)
        return Q / s


def brier_multiclass(P: np.ndarray, y: np.ndarray) -> float:
    Y = np.eye(P.shape[1])[y]
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def multiclass_log_loss(P: np.ndarray, y: np.ndarray) -> float:
    P = np.clip(P, 1e-15, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def _pooled(P: np.ndarray, y: np.ndarray):
    """Flatten to one-vs-rest binary (match,class) pairs for reliability/ECE."""
    return P.ravel(), np.eye(P.shape[1])[y].ravel()


def reliability_table(P: np.ndarray, y: np.ndarray, bins: int = 10) -> list[tuple]:
    p, t = _pooled(P, y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        hi_inclusive = i == bins - 1
        m = (p >= edges[i]) & ((p <= edges[i + 1]) if hi_inclusive else (p < edges[i + 1]))
        if m.sum():
            rows.append((edges[i], edges[i + 1], int(m.sum()),
                         float(p[m].mean()), float(t[m].mean())))
    return rows


def ece(P: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error (pooled one-vs-rest), the single-number summary
    of the reliability table."""
    p, _ = _pooled(P, y)
    n = len(p)
    return float(sum(cnt / n * abs(pred - obs)
                     for _, _, cnt, pred, obs in reliability_table(P, y, bins)))

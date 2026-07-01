"""Learned rerankers for stage-2 ranking (P9).

Two-stage retrieve -> rank is the industry-standard recommendation/search
architecture (2025-2026): cheap signals *retrieve* candidates, then a single
model *learns* to order them from data. This is the documented fix for the
known limitation that fixed-weight fusion (RRF) cannot beat a sharp single
signal — replace fixed weights with weights learned from interactions.
See ADR-0010 and docs/evidence/ltr_industry_standard_2026.md.

This module ships the dependency-free default (``LogisticReranker``, a pointwise
learned combiner, deterministic, runs in CI). The production-grade LambdaMART
lives behind the optional ``[rank]`` extra (``providers/rank_lightgbm.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Feature order shared by every reranker and the candidate builder.
FEATURE_NAMES = ("content", "collab", "tag", "popularity")


def sigmoid(z: float) -> float:
    """Numerically stable logistic function."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


class LogisticReranker:
    """Pointwise logistic-regression reranker — pure Python, deterministic.

    Full-batch gradient descent from a zero init has no randomness, so the same
    training data yields identical weights every run. Features are z-score
    standardized with statistics learned on the training set. This is a *learned*
    combiner (weights come from data), the honest minimal counterpart to the
    fixed-weight RRF hybrid — the same rank-reciprocal features, learned weights.
    """

    name = "logistic"

    def __init__(self, n_iter: int = 500, lr: float = 0.5, l2: float = 1e-4) -> None:
        self.n_iter = n_iter
        self.lr = lr
        self.l2 = l2
        self._mean: list[float] = []
        self._std: list[float] = []
        self._w: list[float] = []
        self._b: float = 0.0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int], groups=None) -> LogisticReranker:
        n = len(X)
        if n == 0:
            raise ValueError("LogisticReranker.fit: empty training set")
        d = len(X[0])
        self._mean = [sum(row[j] for row in X) / n for j in range(d)]
        self._std = []
        for j in range(d):
            var = sum((row[j] - self._mean[j]) ** 2 for row in X) / n
            self._std.append(math.sqrt(var) if var > 1e-12 else 1.0)
        xs = [[(row[j] - self._mean[j]) / self._std[j] for j in range(d)] for row in X]

        self._w = [0.0] * d
        self._b = 0.0
        for _ in range(self.n_iter):
            gw = [0.0] * d
            gb = 0.0
            for xi, yi in zip(xs, y, strict=True):
                z = self._b + sum(self._w[j] * xi[j] for j in range(d))
                err = sigmoid(z) - yi
                for j in range(d):
                    gw[j] += err * xi[j]
                gb += err
            for j in range(d):
                self._w[j] -= self.lr * (gw[j] / n + self.l2 * self._w[j])
            self._b -= self.lr * (gb / n)
        return self

    def predict(self, x: Sequence[float]) -> float:
        d = len(self._w)
        z = self._b + sum(
            self._w[j] * ((x[j] - self._mean[j]) / self._std[j]) for j in range(d)
        )
        return sigmoid(z)

    @property
    def weights(self) -> dict[str, float]:
        """Learned per-signal weights (on standardized features), for inspection."""
        return {FEATURE_NAMES[j]: self._w[j] for j in range(len(self._w))}

"""LightGBM LambdaMART reranker — the production-grade stage-2 model (P9).

LightGBM (Microsoft, MIT) is the de-facto learning-to-rank workhorse in industry
(2025-2026): its ``lambdarank`` objective is LambdaMART, a gradient-boosted-tree
listwise ranker that directly optimizes nDCG. Unlike the linear logistic default,
the boosted trees capture non-linear feature interactions — e.g. "when the
collaborative signal fires, trust it almost exclusively" — which fixed-weight RRF
and a linear combiner cannot express.

Optional: installed via the ``[rank]`` extra. The harness falls back to the
dependency-free ``LogisticReranker`` when LightGBM is unavailable.
See ADR-0010 and docs/evidence/ltr_industry_standard_2026.md.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

# We fit/predict on plain float lists (no column names) by design; silence the
# cosmetic sklearn feature-name check that fires once per single-row predict.
warnings.filterwarnings(
    "ignore", message=".*does not have valid feature names.*", category=UserWarning
)


class LightGBMReranker:
    name = "lightgbm"

    def __init__(
        self, n_estimators: int = 200, num_leaves: int = 15, learning_rate: float = 0.1
    ) -> None:
        self.n_estimators = n_estimators
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self._model = None

    @staticmethod
    def available() -> bool:
        try:
            import lightgbm  # noqa: F401

            return True
        except ImportError:
            return False

    def fit(
        self, X: Sequence[Sequence[float]], y: Sequence[int], groups: Sequence[int]
    ) -> LightGBMReranker:
        import lightgbm as lgb

        if not groups:
            groups = [len(X)]
        self._model = lgb.LGBMRanker(
            objective="lambdarank",
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=5,
            n_jobs=1,
            random_state=42,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        self._model.fit(
            [list(r) for r in X],
            list(y),
            group=list(groups),
            eval_at=[10],
        )
        return self

    def predict(self, x: Sequence[float]) -> float:
        if self._model is None:
            raise RuntimeError("LightGBMReranker.predict before fit")
        return float(self._model.predict([list(x)], validate_features=False)[0])

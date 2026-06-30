"""Offline A-B simulation with a business KPI (R-EVAL-3).

Treats each held-out user as a session. The KPI is hit-rate@k — did a relevant
(held-out) item appear in the variant's top-k — a standard offline proxy for
click-through rate. We report KPI per variant, absolute/relative lift, and a
deterministic bootstrap confidence interval over users for the difference, so
the "win" is qualified by uncertainty rather than asserted from a point estimate.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence


def hit_at_k(
    qrels: Mapping[str, Mapping[str, float]], runs: Mapping[str, Sequence[str]], k: int
) -> dict[str, int]:
    """Per-user 1/0: did any relevant item land in the top-k (proxy click)."""
    out: dict[str, int] = {}
    for user, ranking in runs.items():
        relevant = {d for d, r in qrels.get(user, {}).items() if r >= 1}
        out[user] = int(any(d in relevant for d in list(ranking)[:k]))
    return out


def ab_compare(
    qrels: Mapping[str, Mapping[str, float]],
    runs_a: Mapping[str, Sequence[str]],
    runs_b: Mapping[str, Sequence[str]],
    k: int = 10,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | str]:
    hits_a = hit_at_k(qrels, runs_a, k)
    hits_b = hit_at_k(qrels, runs_b, k)
    users = sorted(set(hits_a) & set(hits_b))
    n = len(users)
    if n == 0:
        return {"n": 0, "kpi_a": 0.0, "kpi_b": 0.0, "abs_lift": 0.0, "decision": "no-data"}

    a = [hits_a[u] for u in users]
    b = [hits_b[u] for u in users]
    kpi_a = sum(a) / n
    kpi_b = sum(b) / n
    abs_lift = kpi_b - kpi_a
    rel_lift = (abs_lift / kpi_a) if kpi_a > 0 else float("inf")

    # paired bootstrap over users (deterministic) for the diff KPI_b - KPI_a
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[min(n_boot - 1, int(0.975 * n_boot))]

    if lo > 0:
        decision = "B wins"
    elif hi < 0:
        decision = "A wins"
    else:
        decision = "inconclusive"

    return {
        "n": n,
        "k": k,
        "kpi_a": kpi_a,
        "kpi_b": kpi_b,
        "abs_lift": abs_lift,
        "rel_lift": rel_lift,
        "ci95_low": lo,
        "ci95_high": hi,
        "decision": decision,
    }

"""Content-based recommender (embedding kNN) — T-28.

The implementation lives in ``baselines.ContentRanker`` (it is also one of the
eval baselines); this module is its named home in the recommender lineup and
re-exports it so call sites can import from either location.
"""

from __future__ import annotations

from recolens.packs.ugc.baselines import ContentRanker

__all__ = ["ContentRanker"]

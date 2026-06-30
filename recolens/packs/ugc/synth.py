"""Synthetic UGC dataset generator (seeded, deterministic).

A recommender benchmark is only worth anything if it *discriminates* between
methods for an honest reason. This generator encodes **two strong, disjoint
signals** so that each single method is good at one half and the hybrid — which
sees both — is the robust winner across seeds and sizes (not a cherry-picked
seed):

1. **Content signal.** Each item has one dominant topic *tag* with a
   tag-correlated body, and each user has one interest tag. A user's interest
   items (and their held-out test items) share that tag, so an embedding/tag
   method recovers them. Collaborative filtering cannot — interest items are
   personal, not co-read across users.
2. **Collaborative signal.** Each user also belongs to a small *cohort* that
   co-reads a shared "signature" set of items drawn independently of tags. Only
   item-item co-occurrence recovers these; content/tags cannot, because the
   signature items don't match the user's interest tag.

Neither single method captures both halves, so Reciprocal-Rank-Fusion of content
+ collaborative + tag (the hybrid) wins robustly — an honest, mechanistic result.
No real/customer data is ever used.
"""

from __future__ import annotations

import random
from typing import Any

# Each tag has a themed vocabulary so item text correlates with its category —
# a fantasy story reads differently from a how-to. Under the default hash
# embedder, shared vocabulary tokens make same-tag items cluster, giving the
# content method genuine, separable signal (not a rigged outcome).
_TAG_VOCAB: dict[str, list[str]] = {
    "fantasy": ["dragon", "wizard", "kingdom", "sword", "elf", "quest", "magic", "castle"],
    "romance": ["love", "heart", "kiss", "wedding", "letter", "longing", "promise", "tears"],
    "scifi": ["spaceship", "android", "galaxy", "quantum", "cyborg", "orbit", "reactor", "alien"],
    "mystery": ["detective", "clue", "murder", "alibi", "suspect", "shadow", "case", "motive"],
    "essay": ["argument", "society", "reflection", "culture", "opinion", "theory", "ethics"],
    "howto": ["step", "guide", "install", "configure", "tutorial", "setup", "tip", "build"],
    "horror": ["ghost", "blood", "haunted", "scream", "curse", "darkness", "monster", "grave"],
    "comedy": ["joke", "laugh", "absurd", "prank", "witty", "silly", "punchline", "banana"],
    "history": ["empire", "ancient", "war", "dynasty", "revolution", "archive", "century"],
    "poetry": ["verse", "rhythm", "metaphor", "stanza", "moonlight", "silence", "echo", "dawn"],
}
TAGS = list(_TAG_VOCAB.keys())
_GLOBAL_NOISE = ["the", "and", "of", "a", "in", "to", "with", "for"]


def _themed_body(rng: random.Random, tags: list[str], n: int) -> str:
    vocab = [w for t in tags for w in _TAG_VOCAB[t]] + _GLOBAL_NOISE
    return " ".join(rng.choice(vocab) for _ in range(n))


def _themed_title(rng: random.Random, tags: list[str], n: int) -> str:
    vocab = [w for t in tags for w in _TAG_VOCAB[t]]
    return " ".join(rng.choice(vocab) for _ in range(n)).title()


def generate(
    n_items: int = 300,
    n_users: int = 80,
    *,
    seed: int = 42,
    cohort_size: int = 4,
    base_ts: int = 1_700_000_000,
) -> dict[str, list[dict[str, Any]]]:
    """Return {'items':[...], 'users':[...], 'interactions':[...]} (dicts)."""
    rng = random.Random(seed)

    # Items have ONE dominant tag (clean clusters → strong, separable content
    # signal). A minority carry a second tag for mild realism/noise.
    items: list[dict[str, Any]] = []
    items_by_tag: dict[str, list[str]] = {t: [] for t in TAGS}
    for i in range(n_items):
        primary = rng.choice(TAGS)
        tags = [primary]
        if rng.random() < 0.2:
            tags.append(rng.choice([t for t in TAGS if t != primary]))
        iid = f"item-{i:04d}"
        items.append(
            {
                "item_id": iid,
                "title": _themed_title(rng, tags, 3),
                "body": _themed_body(rng, tags, 20),
                "tags": tags,
                "author_id": f"author-{rng.randint(0, max(1, n_users // 2)):03d}",
                "created_ts": base_ts + i * 3600,
            }
        )
        items_by_tag[primary].append(iid)

    # Cohort signature sets: collaborative-only signal. Small cohorts with full
    # participation → dense, recoverable co-occurrence. Signatures are random
    # items (tag-independent), so content/tags cannot recover them.
    n_cohorts = max(2, n_users // cohort_size)
    sig_size = 6
    all_ids = [it["item_id"] for it in items]
    rng.shuffle(all_ids)
    cohort_signatures = [
        [all_ids[(c * sig_size + j) % len(all_ids)] for j in range(sig_size)]
        for c in range(n_cohorts)
    ]

    users: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    events = ("view", "read", "like")
    event_weight = {"view": 1.0, "read": 2.0, "like": 3.0}

    for u in range(n_users):
        interest = rng.choice(TAGS)
        cohort = u % n_cohorts
        uid = f"user-{u:04d}"
        users.append(
            {
                "user_id": uid,
                "interests": [interest],
                "cohort": cohort,
                "created_ts": base_ts + u * 60,
            }
        )

        # (1) content signal: items of the user's single interest tag
        pool = sorted(items_by_tag[interest])
        rng.shuffle(pool)
        interest_items = pool[: min(8, len(pool))]

        # (2) collaborative signal: the cohort's full shared signature set
        cohort_items = list(cohort_signatures[cohort])

        history = list(dict.fromkeys(interest_items + cohort_items))
        rng.shuffle(history)  # interleave so both signals land in train AND test
        for k, iid in enumerate(history):
            ev = rng.choices(events, weights=(5, 3, 2))[0]
            interactions.append(
                {
                    "user_id": uid,
                    "item_id": iid,
                    "event": ev,
                    "ts": base_ts + u * 1000 + k,
                    "weight": event_weight[ev],
                }
            )

    return {"items": items, "users": users, "interactions": interactions}

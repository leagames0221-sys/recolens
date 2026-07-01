"""MovieLens 100k loader -> recolens schema dicts (real-data eval).

Maps the classic IR/recsys benchmark into the same ``{"items", "interactions"}``
shape the synthetic generator produces, so the *identical* harness runs on real
data. A rating >= ``POS_THRESHOLD`` is treated as a positive interaction (the
standard implicit-feedback binarization); genres become item tags (the content
signal); the rating timestamp drives the temporal split.

The data is downloaded by ``scripts/fetch_movielens.py`` (not vendored — GroupLens
forbids redistribution). Citation: Harper & Konstan 2015, ACM TiiS 5(4).
"""

from __future__ import annotations

from pathlib import Path

POS_THRESHOLD = 4  # rating >= 4 (of 5) = positive; standard implicit binarization

# u.item genre columns, in file order (README: 19 flags after the first 5 fields).
GENRES = (
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
)


def _default_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "ml-100k"


def load(data_dir: str | Path | None = None) -> dict:
    """Return {"items": [...], "interactions": [...]} from a local ml-100k dir."""
    root = Path(data_dir) if data_dir else _default_dir()
    u_item = root / "u.item"
    u_data = root / "u.data"
    if not u_item.exists() or not u_data.exists():
        raise FileNotFoundError(
            f"MovieLens 100k not found under {root}. "
            "Run `python scripts/fetch_movielens.py` first (it is not vendored)."
        )

    items = []
    # u.item is latin-1; pipe-separated: id|title|release|video|url|<19 genre flags>
    for line in u_item.read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        f = line.split("|")
        flags = f[5:5 + len(GENRES)]
        tags = tuple(
            g for g, on in zip(GENRES, flags, strict=False) if on == "1" and g != "unknown"
        )
        items.append({"item_id": f[0], "title": f[1], "tags": tags})

    interactions = []
    # u.data: user \t item \t rating \t timestamp
    for line in u_data.read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        uid, iid, rating, ts = line.split("\t")
        if int(rating) >= POS_THRESHOLD:
            interactions.append({"user_id": uid, "item_id": iid, "ts": int(ts)})

    return {"items": items, "interactions": interactions}

"""Fetch the MovieLens 100k dataset locally (NOT redistributed in this repo).

MovieLens data may not be redistributed (GroupLens license), so recolens ships
*no* copy of it — this script downloads it on demand into a gitignored ``data/``
dir for local, real-data evaluation. CI never runs this; the synthetic generator
remains the deterministic fixture.

Security: pinned official HTTPS host + SHA-256 verification of the archive before
extraction (supply-chain gate). Usage:

    python scripts/fetch_movielens.py         # download + verify + extract
    python scripts/fetch_movielens.py --print-hash   # just show the archive hash

Citation: F. M. Harper and J. A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM TiiS 5, 4. https://doi.org/10.1145/2827872
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
# SHA-256 of the official ml-100k.zip (verified 2026-07-01). Empty => skip check
# with an explicit warning (first-run bootstrap only).
EXPECTED_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ARCHIVE = DATA_DIR / "ml-100k.zip"
EXTRACTED = DATA_DIR / "ml-100k"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} ...")
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (pinned https host)
        dest.write_bytes(r.read())
    print(f"  -> {dest} ({dest.stat().st_size} bytes)")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-hash", action="store_true", help="download and print the SHA-256 only")
    args = ap.parse_args()

    if not ARCHIVE.exists():
        _download(URL, ARCHIVE)
    digest = _sha256(ARCHIVE)

    if args.print_hash:
        print(f"SHA-256: {digest}")
        return 0

    expected = EXPECTED_SHA256.replace(" ", "")
    if expected and len(expected) == 64:
        if digest != expected:
            print(f"SHA-256 MISMATCH\n  expected {expected}\n  actual   {digest}", file=sys.stderr)
            print("refusing to extract an unverified archive (supply-chain gate).", file=sys.stderr)
            return 2
        print("SHA-256 verified OK")
    else:
        print(f"WARNING: no pinned hash; got {digest}. Set EXPECTED_SHA256 to pin it.")

    with zipfile.ZipFile(ARCHIVE) as z:
        z.extractall(DATA_DIR)
    print(f"extracted -> {EXTRACTED}")
    print("now run:  uv run recolens eval --dataset movielens")
    return 0


if __name__ == "__main__":
    sys.exit(main())

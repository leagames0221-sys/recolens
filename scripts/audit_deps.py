"""License audit (K-1 / R-SEC-2).

Allow only permissive licenses (MIT / Apache-2.0 / BSD / ISC) for any
distribution recolens depends on. Exit non-zero if a disallowed license is
found, so CI can gate it.

Pure stdlib. The classification logic is exposed as functions so tests can
feed negative examples (confirmatory-test 禁止).
"""

from __future__ import annotations

import sys
from importlib import metadata

# Substrings that mark an allowed permissive license.
ALLOWED_MARKERS = ("mit", "apache", "bsd", "isc", "python software foundation", "psf")
# Substrings that are explicitly disallowed (copyleft / proprietary).
DISALLOWED_MARKERS = ("gpl", "agpl", "lgpl", "proprietary", "commercial", "cc-by-nc")


def classify_license(text: str | None) -> str:
    """Return 'allowed' | 'disallowed' | 'unknown' for a license string."""
    if not text:
        return "unknown"
    low = text.lower()
    if any(m in low for m in DISALLOWED_MARKERS):
        return "disallowed"
    if any(m in low for m in ALLOWED_MARKERS):
        return "allowed"
    return "unknown"


def _license_text(dist: metadata.Distribution) -> str:
    """Reliable, short license signals only.

    Trove classifiers (``License :: OSI Approved :: ...``) and the SPDX
    ``License-Expression`` field are reliable identifiers. The free-text
    ``License`` field is used only when short (an identifier, not a full
    license body) — scanning a full BSD/MIT body for substrings causes false
    positives (e.g. SciPy's BSD text), so long bodies are left to 'unknown'.
    """
    md = dist.metadata
    classifiers = [c for c in md.get_all("Classifier", []) if "License ::" in c]
    expr = md.get("License-Expression") or ""
    lic_field = (md.get("License") or "").strip()
    short_lic = lic_field if 0 < len(lic_field) <= 40 else ""
    return " ".join([*classifiers, expr, short_lic]).strip()


def audit_installed() -> tuple[list[str], list[str]]:
    """Return (disallowed, unknown) distribution descriptions."""
    disallowed: list[str] = []
    unknown: list[str] = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name", "?")
        verdict = classify_license(_license_text(dist))
        if verdict == "disallowed":
            disallowed.append(f"{name} ({_license_text(dist)!r})")
        elif verdict == "unknown":
            unknown.append(name)
    return disallowed, unknown


def main() -> int:
    disallowed, unknown = audit_installed()
    if disallowed:
        print("DISALLOWED licenses found (K-1 violation):")
        for d in disallowed:
            print(f"  - {d}")
        return 1
    if unknown:
        print("WARNING: unknown license (manual review needed):")
        for u in sorted(unknown):
            print(f"  - {u}")
    print("license audit OK (no disallowed licenses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

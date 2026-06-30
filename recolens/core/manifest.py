"""Run manifest persistence (MLOps reproducibility, R-OPS-1).

Every run writes config / seed / counts / timing under runs/<ts>/ so results
are reproducible and auditable. The runs/ dir is PRIVATE (gitignored).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def new_run_dir(base: str | Path = "runs", *, now: float | None = None) -> Path:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now if now is not None else time.time()))
    return Path(base) / ts


def write_manifest(run_dir: str | Path, payload: dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

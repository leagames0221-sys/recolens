"""Golden accuracy gates for classify / moderate (T-37, R-EVAL-4 style).

Fixed labeled cases (including injection/spam/benign and an off-topic 'other')
with a deterministic rule layer must classify/moderate exactly. A tampered
expectation must fail (negative case).
"""

from __future__ import annotations

import json
from pathlib import Path

from recolens.packs.ugc.classify import classify
from recolens.packs.ugc.moderate import moderate

GOLDEN = Path(__file__).resolve().parents[1] / "docs" / "golden"


def _load(name):
    lines = (GOLDEN / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_moderation_golden_accuracy():
    cases = _load("moderation_cases.jsonl")
    correct = sum(moderate(c["text"]).action == c["expected_action"] for c in cases)
    assert correct == len(cases), f"{correct}/{len(cases)} moderation cases correct"


def test_classification_golden_accuracy():
    cases = _load("classification_cases.jsonl")
    correct = sum(classify(c["text"]).category == c["expected_category"] for c in cases)
    assert correct == len(cases), f"{correct}/{len(cases)} classification cases correct"


def test_moderation_golden_gate_catches_tampering():
    # flipping an expected label must break the gate (negative case)
    cases = _load("moderation_cases.jsonl")
    tampered = dict(cases[0])
    tampered["expected_action"] = "allow"  # it is actually 'block'
    assert moderate(tampered["text"]).action != tampered["expected_action"]

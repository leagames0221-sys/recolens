"""Phase 0 smoke tests: CLI wiring + license audit (with negative cases)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audit_deps  # noqa: E402

from recolens.cli import SUBCOMMANDS, build_parser, main  # noqa: E402


def test_version_runs():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_no_command_prints_help_and_returns_zero():
    assert main([]) == 0


def test_all_subcommands_parse():
    parser = build_parser()
    # some subcommands take required positional args
    extra_args = {"search": ["q"], "classify": ["t"], "moderate": ["t"]}
    for name, _ in SUBCOMMANDS:
        ns = parser.parse_args([name, *extra_args.get(name, [])])
        assert ns.command == name
        assert callable(ns._handler)


# --- license audit: positive AND negative examples (confirmatory-test 禁止) ---


@pytest.mark.parametrize(
    "text",
    ["MIT", "Apache Software License", "BSD-3-Clause", "ISC License", "Apache-2.0"],
)
def test_allowed_licenses(text):
    assert audit_deps.classify_license(text) == "allowed"


@pytest.mark.parametrize(
    "text",
    ["GPL-3.0", "AGPL", "LGPL-2.1", "Proprietary", "CC-BY-NC-4.0"],
)
def test_disallowed_licenses_are_rejected(text):
    # 落ちるべき入力で確実に 'disallowed' を返すこと(負例)
    assert audit_deps.classify_license(text) == "disallowed"


def test_unknown_license():
    assert audit_deps.classify_license(None) == "unknown"
    assert audit_deps.classify_license("Some Custom EULA") == "unknown"

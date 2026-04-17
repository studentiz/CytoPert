#!/usr/bin/env python3
"""Fail when CJK characters appear in CytoPert source code.

CytoPert is an English-only project. The previous policy whitelisted
``tests/manual/run_deepseek_live.py`` and ``README.zh-CN.md`` so live
tests could send Chinese prompts and a Chinese README could ship next
to the English one. Both have been removed; the only remaining
whitelist entries are:

* ``references/**`` -- vendored upstream sources (Hermes-agent etc.)
  that we keep verbatim for diff transparency. Any CJK in them is
  produced by upstream and is not ours to translate.
* ``cytopert/skills/bundled/**`` -- bundled SKILL.md sheets MAY use
  CJK in worked examples (the user can pin the language). The CytoPert
  ones currently do not, but the policy stays permissive in case a
  future skill ships in another language.

Run as ``python scripts/check_no_chinese.py [path ...]`` (default scans
``cytopert/`` and ``tests/`` excluding the whitelist). Exit code 0 on a
clean tree; 1 with a per-file:line listing on a violation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CJK_RANGE = re.compile(r"[\u4e00-\u9fff]")

#: Per-file allow-list (no CJK here today; kept intentionally empty so
#: future additions are reviewed explicitly).
WHITELIST_SUFFIXES: tuple[str, ...] = ()
WHITELIST_PREFIXES = (
    "cytopert/skills/bundled/",
    "references/",
)


def _is_whitelisted(rel: Path) -> bool:
    rel_str = rel.as_posix()
    if rel_str in WHITELIST_SUFFIXES:
        return True
    return any(rel_str.startswith(prefix) for prefix in WHITELIST_PREFIXES)


def _scan_file(path: Path, root: Path) -> list[tuple[int, str]]:
    rel = path.relative_to(root)
    if _is_whitelisted(rel):
        return []
    if path.suffix not in {".py", ".md", ".toml", ".yaml", ".yml", ".json"}:
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if CJK_RANGE.search(line):
            hits.append((lineno, line.rstrip()))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["cytopert", "tests", "docs", "README.md", "scripts"],
        help=(
            "Paths to scan (defaults to cytopert/, tests/, docs/, "
            "README.md, scripts/)."
        ),
    )
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()

    violations: list[tuple[Path, int, str]] = []
    for raw in args.paths:
        base = (root / raw).resolve()
        if not base.exists():
            continue
        if base.is_file():
            files = [base]
        else:
            files = [p for p in base.rglob("*") if p.is_file()]
        for f in files:
            for lineno, line in _scan_file(f, root):
                violations.append((f.relative_to(root), lineno, line))

    if violations:
        print(
            f"check_no_chinese: {len(violations)} CJK character occurrences "
            f"outside the whitelist:",
            file=sys.stderr,
        )
        for rel, lineno, line in violations:
            print(f"  {rel}:{lineno}: {line[:140]}", file=sys.stderr)
        print(
            "If a file legitimately contains CJK characters (e.g. test "
            "prompts, bundled SKILL.md), add it to WHITELIST_SUFFIXES or "
            "WHITELIST_PREFIXES at the top of scripts/check_no_chinese.py.",
            file=sys.stderr,
        )
        return 1
    print("check_no_chinese: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

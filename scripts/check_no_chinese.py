#!/usr/bin/env python3
"""Fail when CJK characters appear in CytoPert source code.

CytoPert is an English-source project. A handful of files are explicitly
allowed to contain CJK characters:

* ``tests/manual/run_deepseek_live.py`` -- the live test harness sends
  Chinese prompts to DeepSeek to verify model robustness across
  languages.
* ``cytopert/skills/bundled/**`` -- bundled SKILL.md sheets are markdown
  prose that may include CJK examples.
* ``docs/**`` -- documentation (the Chinese README in particular).
* ``references/**`` -- vendored upstream sources (we do not modify them).
* ``README.zh-CN.md`` -- the Chinese README.

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

#: Glob suffixes whose hits are silently allowed.
WHITELIST_SUFFIXES = (
    "tests/manual/run_deepseek_live.py",
    "README.zh-CN.md",
)
WHITELIST_PREFIXES = (
    "cytopert/skills/bundled/",
    "docs/",
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
        default=["cytopert", "tests"],
        help="Paths to scan (defaults to cytopert/ and tests/).",
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

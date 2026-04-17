"""Skills hub: install skills from local paths, archives, or git URLs.

The hub is intentionally thin -- the SkillsManager already owns the
on-disk layout and validation; this module just resolves a remote /
archive source into a local SKILL.md tree and forwards it to the
manager.

Supported source forms (auto-detected):

* A directory containing ``SKILL.md`` (and optionally supplementary
  files). Copied verbatim under ``<skills_dir>/<category>/<name>/``.
* A ``.zip`` / ``.tar.gz`` / ``.tgz`` archive whose root holds
  ``SKILL.md`` (or a single subdirectory containing it).
* A git URL (any path ending in ``.git`` or starting with
  ``https://github.com/``, ``https://gitlab.com/``,
  ``git@``). Cloned shallowly into a tempdir, then treated as a local
  directory.

Failure modes raise ``ValueError`` (bad source) or ``FileNotFoundError``
(no SKILL.md) so the CLI layer can convert them into typer errors.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from cytopert.skills.manager import SkillsManager


_GIT_PREFIXES = ("https://github.com/", "https://gitlab.com/", "git@")
_ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz")


def _is_git_url(source: str) -> bool:
    return source.endswith(".git") or source.startswith(_GIT_PREFIXES)


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(s) for s in _ARCHIVE_SUFFIXES)


def _resolve_skill_root(folder: Path) -> Path:
    """Find the directory containing SKILL.md inside an unpacked source."""
    if (folder / "SKILL.md").is_file():
        return folder
    # Single-subdirectory archives (e.g. `cool-skill-main/`) -- step in.
    children = [p for p in folder.iterdir() if p.is_dir()]
    if len(children) == 1 and (children[0] / "SKILL.md").is_file():
        return children[0]
    # Search up to two levels deep for SKILL.md.
    for child in folder.rglob("SKILL.md"):
        rel_depth = len(child.relative_to(folder).parts)
        if rel_depth <= 3:  # SKILL.md, name/SKILL.md, repo/name/SKILL.md
            return child.parent
    raise FileNotFoundError(
        f"Source under {folder} does not contain a SKILL.md "
        "(searched up to two levels deep)."
    )


def _git_clone(url: str, dest: Path) -> None:
    """Shallow clone ``url`` into ``dest``; raises ValueError on failure."""
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "git clone failed: the `git` binary was not found on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"git clone failed (exit {exc.returncode}): {exc.stderr.strip()}"
        ) from exc


def _extract_archive(path: Path, dest: Path) -> None:
    name = path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
    elif name.endswith((".tar.gz", ".tgz")):
        # ``filter='data'`` (added in 3.12) prevents tarballs from
        # writing outside the destination tree; fall back gracefully on
        # older Pythons by checking each member's path manually.
        with tarfile.open(path, "r:gz") as t:
            try:
                t.extractall(dest, filter="data")
            except TypeError:
                _safe_tar_extract(t, dest)
    else:
        raise ValueError(f"Unsupported archive type: {path.name}")


def _safe_tar_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract ``tar`` into ``dest`` rejecting paths that escape the root."""
    base = dest.resolve()
    for member in tar.getmembers():
        target = (base / member.name).resolve()
        if not str(target).startswith(str(base)):
            raise ValueError(f"Archive escape attempt: {member.name!r}")
    tar.extractall(dest)


def install_from_source(
    manager: "SkillsManager",
    *,
    source: str,
    name: str | None = None,
    category: str = "user",
    force: bool = False,
) -> Path:
    """Install a skill from ``source`` and return the installed SKILL.md path."""
    if not source:
        raise ValueError("source is required")

    with tempfile.TemporaryDirectory(prefix="cytopert_skill_") as tmp_str:
        tmp = Path(tmp_str)
        unpack_dir = tmp / "unpack"
        unpack_dir.mkdir()

        if _is_git_url(source):
            _git_clone(source, unpack_dir)
            skill_root = _resolve_skill_root(unpack_dir)
        else:
            src_path = Path(source).expanduser()
            if not src_path.exists():
                raise FileNotFoundError(f"No such path: {src_path}")
            if src_path.is_dir():
                skill_root = _resolve_skill_root(src_path)
            elif _is_archive(src_path):
                _extract_archive(src_path, unpack_dir)
                skill_root = _resolve_skill_root(unpack_dir)
            else:
                raise ValueError(
                    f"Unsupported source: {source} "
                    "(must be a directory, .zip / .tar.gz archive, or git URL)"
                )

        target_name = name or skill_root.name
        target_dir = manager.skills_dir / category / target_name
        if target_dir.exists():
            if not force:
                raise ValueError(
                    f"Skill already installed at {target_dir}; pass --force to overwrite."
                )
            shutil.rmtree(target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(skill_root, target_dir)
        return target_dir / "SKILL.md"


__all__ = ["install_from_source"]

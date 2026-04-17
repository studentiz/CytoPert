"""SkillsManager: scan / parse / mutate SKILL.md sheets.

Each skill lives at ``<skills_dir>/<category>/<name>/SKILL.md`` and follows the
[agentskills.io](https://agentskills.io) format with an additional
``metadata.cytopert`` block for category and tool-availability hints.

Skills are surfaced to the LLM via progressive disclosure:

- Level 0 (`render_index`): name + description + category for every skill.
- Level 1 (`view`): full SKILL.md content.
- Level 2 (`view_file`): supplementary file inside the skill directory.

Agent-proposed skills go to ``<skills_dir>/.staged/<name>/SKILL.md`` first;
the user promotes them with ``cytopert skills accept <name>``.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

SKILL_FILENAME = "SKILL.md"
STAGED_DIR_NAME = ".staged"
BUNDLED_MANIFEST = ".bundled_manifest"

_SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


@dataclass
class SkillMeta:
    """Skill metadata parsed from SKILL.md frontmatter."""

    name: str
    description: str = ""
    version: str = "0.1.0"
    category: str = "uncategorized"
    tags: list[str] = field(default_factory=list)
    requires_tools: list[str] = field(default_factory=list)
    fallback_for_tools: list[str] = field(default_factory=list)
    path: Path | None = None
    staged: bool = False
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "tags": self.tags,
            "requires_tools": self.requires_tools,
            "fallback_for_tools": self.fallback_for_tools,
            "staged": self.staged,
            "path": str(self.path) if self.path else None,
        }


class SkillsManager:
    """Filesystem-backed registry of SKILL.md sheets."""

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = Path(skills_dir)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.staged_dir = self.skills_dir / STAGED_DIR_NAME
        self.staged_dir.mkdir(parents=True, exist_ok=True)

    def install_bundled(self, force: bool = False) -> int:
        """Copy bundled SKILL.md sheets into the user skills dir on first run.

        The manifest file is only written after at least one skill is copied
        successfully. The legacy implementation stamped the manifest even when
        ``copied == 0`` (e.g. because the bundled package data was missing),
        which caused subsequent runs to skip installation forever and the
        researcher to see "no bundled skills" with no way to recover short of
        ``rm ~/.cytopert/skills/.bundled_manifest``.
        """
        manifest = self.skills_dir / BUNDLED_MANIFEST
        if manifest.exists() and not force:
            return 0
        copied = 0
        try:
            bundled_root = resources.files("cytopert.skills.bundled")
        except (ModuleNotFoundError, FileNotFoundError) as exc:
            logger.warning("Bundled skills package data unavailable: %s", exc)
            return 0
        for category_dir in bundled_root.iterdir():
            if not category_dir.is_dir() or category_dir.name.startswith("_"):
                continue
            for skill_dir in category_dir.iterdir():
                if not skill_dir.is_dir():
                    continue
                skill_path = skill_dir / SKILL_FILENAME
                if not skill_path.is_file():
                    continue
                target_dir = self.skills_dir / category_dir.name / skill_dir.name
                if target_dir.exists() and not force:
                    continue
                target_dir.mkdir(parents=True, exist_ok=True)
                target_dir.joinpath(SKILL_FILENAME).write_text(
                    skill_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
                copied += 1
        if copied > 0 or force:
            manifest.write_text(f"installed bundled skills: {copied}\n", encoding="utf-8")
        else:
            logger.warning(
                "install_bundled copied 0 skills; not stamping manifest so the next "
                "run will retry. Check that cytopert.skills.bundled is on the "
                "wheel's package data."
            )
        return copied

    def list(self, include_staged: bool = False) -> list[SkillMeta]:
        out: list[SkillMeta] = []
        for skill_path in self._iter_skill_files(self.skills_dir, skip_staged=True):
            meta = self._parse_skill_file(skill_path, staged=False)
            if meta:
                out.append(meta)
        if include_staged:
            for skill_path in self._iter_skill_files(self.staged_dir, skip_staged=False):
                meta = self._parse_skill_file(skill_path, staged=True)
                if meta:
                    out.append(meta)
        out.sort(key=lambda s: (s.category, s.name))
        return out

    def view(self, name: str, include_staged: bool = True) -> str:
        path = self._find_skill_path(name, include_staged=include_staged)
        if not path:
            raise FileNotFoundError(f"Skill {name!r} not found")
        return path.read_text(encoding="utf-8")

    def view_file(self, name: str, rel: str) -> str:
        skill_dir = self._find_skill_dir(name)
        if not skill_dir:
            raise FileNotFoundError(f"Skill {name!r} not found")
        rel_norm = rel.lstrip("/")
        target = (skill_dir / rel_norm).resolve()
        if skill_dir.resolve() not in target.parents and target != skill_dir / SKILL_FILENAME:
            raise PermissionError("Refusing to read outside the skill directory")
        if not target.is_file():
            raise FileNotFoundError(f"File {rel!r} not found in skill {name!r}")
        return target.read_text(encoding="utf-8")

    def create(
        self,
        name: str,
        content: str,
        category: str | None = None,
        staged: bool = False,
    ) -> Path:
        self._validate_name(name)
        target_root = self.staged_dir if staged else self.skills_dir
        if category and not staged:
            target_dir = target_root / category / name
        else:
            target_dir = target_root / name
        if target_dir.exists():
            raise FileExistsError(f"Skill {name!r} already exists at {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / SKILL_FILENAME
        path.write_text(self._ensure_frontmatter(name, content, category), encoding="utf-8")
        return path

    def patch(self, name: str, old_string: str, new_string: str) -> Path:
        path = self._find_skill_path(name)
        if not path:
            raise FileNotFoundError(f"Skill {name!r} not found")
        text = path.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            raise ValueError("old_string not found")
        if count > 1:
            raise ValueError(f"old_string matched {count} times; provide more unique context")
        path.write_text(text.replace(old_string, new_string, 1), encoding="utf-8")
        return path

    def edit(self, name: str, content: str) -> Path:
        path = self._find_skill_path(name)
        if not path:
            raise FileNotFoundError(f"Skill {name!r} not found")
        path.write_text(content, encoding="utf-8")
        return path

    def delete(self, name: str) -> None:
        skill_dir = self._find_skill_dir(name)
        if not skill_dir:
            raise FileNotFoundError(f"Skill {name!r} not found")
        shutil.rmtree(skill_dir)

    def write_file(self, name: str, rel: str, content: str) -> Path:
        skill_dir = self._find_skill_dir(name)
        if not skill_dir:
            raise FileNotFoundError(f"Skill {name!r} not found")
        rel_norm = rel.lstrip("/")
        target = (skill_dir / rel_norm).resolve()
        if skill_dir.resolve() not in target.parents:
            raise PermissionError("Refusing to write outside the skill directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def accept_staged(self, name: str, category: str | None = None) -> Path:
        """Promote a staged skill to the live skills directory."""
        staged_path = self._find_skill_path(name, include_staged=True, only_staged=True)
        if not staged_path:
            raise FileNotFoundError(f"No staged skill named {name!r}")
        meta = self._parse_skill_file(staged_path, staged=True)
        cat = category or (meta.category if meta else "uncategorized")
        target_dir = self.skills_dir / cat / name
        if target_dir.exists():
            raise FileExistsError(f"Live skill already exists at {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        for child in staged_path.parent.iterdir():
            if child.is_file():
                shutil.copy2(child, target_dir / child.name)
            elif child.is_dir():
                shutil.copytree(child, target_dir / child.name)
        shutil.rmtree(staged_path.parent)
        return target_dir / SKILL_FILENAME

    def render_index(self, available_tools: set[str] | None = None) -> str:
        skills = self.list(include_staged=False)
        if not skills:
            return "(no skills installed; use skill_manage(action='create', ...) to add one)"
        lines = []
        for s in skills:
            if not self._skill_visible(s, available_tools):
                continue
            lines.append(f"- [{s.category}] {s.name} — {s.description}")
        return "\n".join(lines) if lines else "(no applicable skills for this session)"

    @staticmethod
    def _skill_visible(skill: SkillMeta, available_tools: set[str] | None) -> bool:
        if available_tools is None:
            return True
        if skill.requires_tools:
            if not set(skill.requires_tools).issubset(available_tools):
                return False
        if skill.fallback_for_tools:
            if set(skill.fallback_for_tools).issubset(available_tools):
                return False
        return True

    def _iter_skill_files(self, root: Path, *, skip_staged: bool) -> list[Path]:
        if not root.exists():
            return []
        results: list[Path] = []
        for p in root.rglob(SKILL_FILENAME):
            if skip_staged and STAGED_DIR_NAME in p.parts:
                continue
            results.append(p)
        return results

    def _find_skill_path(
        self,
        name: str,
        include_staged: bool = True,
        only_staged: bool = False,
    ) -> Path | None:
        roots: list[Path]
        if only_staged:
            roots = [self.staged_dir]
        else:
            roots = [self.skills_dir]
            if include_staged:
                roots.append(self.staged_dir)
        for root in roots:
            for p in self._iter_skill_files(root, skip_staged=root is self.skills_dir):
                if p.parent.name == name:
                    return p
        return None

    def _find_skill_dir(self, name: str) -> Path | None:
        path = self._find_skill_path(name)
        return path.parent if path else None

    def _parse_skill_file(self, path: Path, *, staged: bool) -> SkillMeta | None:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read SKILL.md at %s: %s", path, exc)
            return None
        meta_dict, _ = parse_frontmatter(text)
        if not meta_dict:
            return SkillMeta(name=path.parent.name, description="(no frontmatter)", path=path,
                             staged=staged)
        cytopert_meta = (
            meta_dict.get("metadata", {}).get("cytopert", {})
            if isinstance(meta_dict.get("metadata"), dict)
            else {}
        )
        if not isinstance(cytopert_meta, dict):
            cytopert_meta = {}
        category = cytopert_meta.get("category") or self._guess_category(path)
        return SkillMeta(
            name=str(meta_dict.get("name") or path.parent.name),
            description=str(meta_dict.get("description") or ""),
            version=str(meta_dict.get("version") or "0.1.0"),
            category=str(category or "uncategorized"),
            tags=list(cytopert_meta.get("tags") or []),
            requires_tools=list(cytopert_meta.get("requires_tools") or []),
            fallback_for_tools=list(cytopert_meta.get("fallback_for_tools") or []),
            path=path,
            staged=staged,
            raw_metadata=meta_dict,
        )

    def _guess_category(self, path: Path) -> str:
        try:
            rel = path.relative_to(self.skills_dir)
        except ValueError:
            try:
                rel = path.relative_to(self.staged_dir)
            except ValueError:
                return "uncategorized"
        parts = rel.parts
        if len(parts) >= 3:
            return parts[0]
        return "uncategorized"

    def _ensure_frontmatter(self, name: str, content: str, category: str | None) -> str:
        meta_dict, body = parse_frontmatter(content)
        if not meta_dict:
            meta_dict = {"name": name, "description": "(provide a description)"}
        meta_dict.setdefault("name", name)
        meta_dict.setdefault("description", "(provide a description)")
        meta_dict.setdefault("version", "0.1.0")
        cyto = meta_dict.setdefault("metadata", {}).setdefault("cytopert", {})
        if category and not cyto.get("category"):
            cyto["category"] = category
        rebuilt = "---\n" + yaml.safe_dump(meta_dict, sort_keys=False, allow_unicode=True) + "---\n"
        if not body:
            body = f"# {meta_dict['name']}\n\n## When to Use\n\n## Procedure\n\n## Pitfalls\n\n## Verification\n"
        return rebuilt + body

    @staticmethod
    def _validate_name(name: str) -> None:
        if not _SKILL_NAME_RE.match(name):
            raise ValueError(
                f"Invalid skill name {name!r}: use lowercase letters, digits, dashes/underscores only"
            )


_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter and return (metadata dict, body text)."""
    if not text:
        return {}, ""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw_meta, body = m.group(1), m.group(2)
    try:
        loaded = yaml.safe_load(raw_meta) or {}
    except yaml.YAMLError:
        return {}, text
    if not isinstance(loaded, dict):
        return {}, text
    return loaded, body

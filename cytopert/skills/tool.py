"""LLM-facing skill tools: skills_list, skill_view, skill_manage."""

from __future__ import annotations

import json
from typing import Any

from cytopert.agent.tools.base import Tool
from cytopert.skills.manager import SkillsManager


class SkillsListTool(Tool):
    """Return the Level-0 index of installed skills (name + description + category)."""

    def __init__(self, manager: SkillsManager) -> None:
        self.manager = manager

    @property
    def name(self) -> str:
        return "skills_list"

    @property
    def description(self) -> str:
        return (
            "List all installed CytoPert skills (Level 0 index). Returns name, description, "
            "category, and tags only. Use skill_view to load a skill's full content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "include_staged": {
                    "type": "boolean",
                    "description": "Include skills proposed by the agent but not yet accepted.",
                    "default": False,
                },
            },
            "required": [],
        }

    async def execute(self, include_staged: bool = False) -> str:
        skills = self.manager.list(include_staged=include_staged)
        return json.dumps([s.to_dict() for s in skills], ensure_ascii=False)


class SkillViewTool(Tool):
    """Return the full SKILL.md (Level 1) or a specific reference file (Level 2)."""

    def __init__(self, manager: SkillsManager) -> None:
        self.manager = manager

    @property
    def name(self) -> str:
        return "skill_view"

    @property
    def description(self) -> str:
        return (
            "Load a skill's full content. Provide name (Level 1) or name + path (Level 2: a "
            "supplementary file inside the skill directory)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name (e.g. perturbation-de)."},
                "path": {
                    "type": "string",
                    "description": "Optional relative path inside the skill dir for Level 2 view.",
                },
            },
            "required": ["name"],
        }

    async def execute(self, name: str, path: str | None = None) -> str:
        try:
            if path:
                return self.manager.view_file(name, path)
            return self.manager.view(name)
        except FileNotFoundError as e:
            return f"Error: {e}"
        except PermissionError as e:
            return f"Error: {e}"


class SkillManageTool(Tool):
    """Create / patch / edit / delete / write_file / accept skills (procedural memory mutation)."""

    def __init__(self, manager: SkillsManager) -> None:
        self.manager = manager

    @property
    def name(self) -> str:
        return "skill_manage"

    @property
    def description(self) -> str:
        return (
            "Mutate the procedural memory store (skills). Actions: create | patch | edit | delete | "
            "write_file | accept_staged. Newly created skills go to .staged/ by default; the user must "
            "promote them with `cytopert skills accept <name>` (or accept_staged action)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "patch", "edit", "delete", "write_file", "accept_staged"],
                },
                "name": {"type": "string"},
                "content": {"type": "string", "description": "For create/edit: full SKILL.md content."},
                "category": {"type": "string", "description": "Category folder (pipelines, reasoning, ...)."},
                "old_string": {"type": "string", "description": "For patch."},
                "new_string": {"type": "string", "description": "For patch."},
                "file_path": {"type": "string", "description": "For write_file: relative path inside skill dir."},
                "file_content": {"type": "string", "description": "For write_file: file body."},
                "staged": {
                    "type": "boolean",
                    "description": "For create: stage instead of writing live (default true).",
                    "default": True,
                },
            },
            "required": ["action", "name"],
        }

    async def execute(
        self,
        action: str,
        name: str,
        content: str | None = None,
        category: str | None = None,
        old_string: str | None = None,
        new_string: str | None = None,
        file_path: str | None = None,
        file_content: str | None = None,
        staged: bool = True,
    ) -> str:
        try:
            if action == "create":
                if not content:
                    return json.dumps({"success": False, "error": "content required for create"})
                path = self.manager.create(name, content, category=category, staged=staged)
                return json.dumps({"success": True, "path": str(path), "staged": staged})
            if action == "patch":
                if old_string is None or new_string is None:
                    return json.dumps({"success": False, "error": "old_string and new_string required"})
                path = self.manager.patch(name, old_string, new_string)
                return json.dumps({"success": True, "path": str(path)})
            if action == "edit":
                if not content:
                    return json.dumps({"success": False, "error": "content required for edit"})
                path = self.manager.edit(name, content)
                return json.dumps({"success": True, "path": str(path)})
            if action == "delete":
                self.manager.delete(name)
                return json.dumps({"success": True})
            if action == "write_file":
                if not file_path or file_content is None:
                    return json.dumps({"success": False, "error": "file_path and file_content required"})
                path = self.manager.write_file(name, file_path, file_content)
                return json.dumps({"success": True, "path": str(path)})
            if action == "accept_staged":
                path = self.manager.accept_staged(name, category=category)
                return json.dumps({"success": True, "path": str(path)})
            return json.dumps({"success": False, "error": f"Unknown action {action!r}"})
        except (FileNotFoundError, FileExistsError, PermissionError, ValueError) as e:
            return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)

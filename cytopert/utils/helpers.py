"""Helper utilities for CytoPert."""

import os
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path() -> Path:
    """Get the CytoPert data directory (CYTOPERT_HOME or ~/.cytopert)."""
    override = os.environ.get("CYTOPERT_HOME")
    base = Path(override).expanduser() if override else Path.home() / ".cytopert"
    return ensure_dir(base)


def get_workspace_path(config_workspace: str | None = None) -> Path:
    """Get the workspace path, creating it if needed."""
    default = get_data_path() / "workspace"
    path = Path(config_workspace).expanduser().resolve() if config_workspace else default
    return ensure_dir(path)


def get_state_db_path() -> Path:
    """SQLite state DB path (evidence + chains)."""
    return get_data_path() / "state.db"


def get_memory_dir() -> Path:
    """Directory holding CONTEXT.md, RESEARCHER.md, HYPOTHESIS_LOG.md."""
    return ensure_dir(get_data_path() / "memory")


def get_skills_dir() -> Path:
    """Directory holding SKILL.md files (~/.cytopert/skills)."""
    return ensure_dir(get_data_path() / "skills")


def get_chains_dir() -> Path:
    """Directory holding mechanism chain JSONL audit trails."""
    return ensure_dir(get_data_path() / "chains")


def safe_filename(name: str) -> str:
    """Convert a string to a safe filename."""
    unsafe = '<>:"/\\|?*'
    for char in unsafe:
        name = name.replace(char, "_")
    return name.strip()

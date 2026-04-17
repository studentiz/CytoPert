"""Helper utilities for CytoPert.

The CytoPert data directory resolves in this priority order (stage 12):
    1. ``CYTOPERT_HOME`` env var (explicit per-process override).
    2. ``~/.cytopert/active_profile`` file (named profile selection set
       by ``cytopert profile use <name>``); the profile root is then
       ``~/.cytopert/profiles/<name>/``.
    3. ``~/.cytopert/`` (the default root).
"""

import os
from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


CYTOPERT_ROOT_DIR = Path.home() / ".cytopert"
ACTIVE_PROFILE_FILE = "active_profile"
PROFILES_SUBDIR = "profiles"


def _read_active_profile_name() -> str | None:
    """Read the persisted active profile name, or None if no file is set."""
    path = CYTOPERT_ROOT_DIR / ACTIVE_PROFILE_FILE
    if not path.is_file():
        return None
    try:
        name = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return name or None


def get_data_path() -> Path:
    """Resolve the CytoPert data directory honouring the profile system."""
    override = os.environ.get("CYTOPERT_HOME")
    if override:
        return ensure_dir(Path(override).expanduser())
    profile = _read_active_profile_name()
    if profile:
        return ensure_dir(CYTOPERT_ROOT_DIR / PROFILES_SUBDIR / profile)
    return ensure_dir(CYTOPERT_ROOT_DIR)


def profiles_dir() -> Path:
    """Directory holding all named profiles (always at ``~/.cytopert/profiles``)."""
    return ensure_dir(CYTOPERT_ROOT_DIR / PROFILES_SUBDIR)


def active_profile_name() -> str | None:
    """Return the active profile name (env var > active_profile file > None)."""
    override = os.environ.get("CYTOPERT_HOME")
    if override:
        # The env-var override may itself point at a profile dir; report
        # the basename so the doctor / status command can show a useful
        # label even when the user picked a profile via -p.
        path = Path(override).expanduser()
        if path.parent == CYTOPERT_ROOT_DIR / PROFILES_SUBDIR:
            return path.name
        return None
    return _read_active_profile_name()


def set_active_profile(name: str | None) -> None:
    """Persist the active profile name; pass None to clear it."""
    CYTOPERT_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    path = CYTOPERT_ROOT_DIR / ACTIVE_PROFILE_FILE
    if name is None:
        if path.exists():
            path.unlink()
        return
    path.write_text(f"{name}\n", encoding="utf-8")


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

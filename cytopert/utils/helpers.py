"""Helper utilities for CytoPert."""

from pathlib import Path


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path() -> Path:
    """Get the CytoPert data directory (~/.cytopert)."""
    return ensure_dir(Path.home() / ".cytopert")


def get_workspace_path(config_workspace: str | None = None) -> Path:
    """Get the workspace path, creating it if needed."""
    default = Path.home() / ".cytopert" / "workspace"
    path = Path(config_workspace or default).expanduser().resolve()
    return ensure_dir(path)


def safe_filename(name: str) -> str:
    """Convert a string to a safe filename."""
    unsafe = '<>:"/\\|?*'
    for char in unsafe:
        name = name.replace(char, "_")
    return name.strip()

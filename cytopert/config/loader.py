"""Configuration loading utilities for CytoPert."""

import json
from pathlib import Path
from typing import Any

from cytopert.config.schema import Config


def get_config_path() -> Path:
    """Get the default configuration file path (respects CYTOPERT_HOME)."""
    from cytopert.utils.helpers import get_data_path

    return get_data_path() / "config.json"


def get_data_dir() -> Path:
    """Get the CytoPert data directory."""
    from cytopert.utils.helpers import get_data_path
    return get_data_path()


def _convert_keys(data: Any) -> Any:
    """Convert camelCase keys to snake_case for Pydantic."""
    if isinstance(data, dict):
        return {_camel_to_snake(k): _convert_keys(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_convert_keys(item) for item in data]
    return data


def _convert_to_camel(data: Any) -> Any:
    """Convert snake_case keys to camelCase for JSON."""
    if isinstance(data, dict):
        return {_snake_to_camel(k): _convert_to_camel(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_convert_to_camel(item) for item in data]
    return data


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()
    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            return Config.model_validate(_convert_keys(data))
        except (json.JSONDecodeError, ValueError) as e:
            import sys
            print(f"Warning: Failed to load config from {path}: {e}", file=sys.stderr)
            print("Using default configuration.", file=sys.stderr)
    return Config()


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump()
    data = _convert_to_camel(data)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

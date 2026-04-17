"""Bundled workflow scenarios.

Importing this package eagerly imports every same-package module so each
scenario can call ``register_scenario`` from its top level. New
contributions need only drop a new module here and call
``register_scenario(...)`` at the bottom -- no registry edits required.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)


def _autoimport_scenarios() -> list[str]:
    """Import every sibling module so its ``register_scenario`` runs."""
    imported: list[str] = []
    for module_info in pkgutil.iter_modules(__path__):
        if module_info.name.startswith("_"):
            continue
        full_name = f"{__name__}.{module_info.name}"
        try:
            importlib.import_module(full_name)
            imported.append(full_name)
        except Exception as exc:
            logger.warning("Could not import scenario module %s: %s", full_name, exc)
    return imported


_autoimport_scenarios()

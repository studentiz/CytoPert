"""Health-check command for CytoPert installs.

``cytopert doctor`` runs a fixed set of probes (config / API key /
workspace writability / state.db open / FTS5 trigger / bundled SKILLs /
plugin discovery / scanpy import / decoupler import / pathway_lookup
smoke). Each probe prints PASS / WARN / FAIL plus a one-line hint.

Returns 0 when there are no FAIL rows; 1 otherwise. The optional
``--ping`` flag adds a 1-token LLM round-trip; off by default to keep
``doctor`` invocations free.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path

from rich.console import Console
from rich.table import Table


def _row(level: str, name: str, detail: str) -> tuple[str, str, str]:
    return (level, name, detail)


def _check_profile() -> tuple[str, str, str]:
    from cytopert.utils.helpers import active_profile_name, get_data_path

    name = active_profile_name()
    home = get_data_path()
    if name:
        return _row("PASS", "profile", f"{name} (root={home})")
    return _row("PASS", "profile", f"default root ({home})")


def _check_config() -> tuple[str, str, str]:
    from cytopert.config.loader import get_config_path, load_config

    path = get_config_path()
    if not path.exists():
        return _row(
            "FAIL",
            "config.json",
            f"No file at {path}; run `cytopert setup`.",
        )
    try:
        load_config()
    except Exception as exc:  # noqa: BLE001
        return _row("FAIL", "config.json", f"parse failed: {exc}")
    return _row("PASS", "config.json", str(path))


def _check_provider() -> tuple[str, str, str]:
    from cytopert.config.loader import load_config

    cfg = load_config()
    provider = cfg.get_provider_type()
    if not provider:
        return _row(
            "WARN",
            "provider",
            "No provider key configured; run `cytopert setup` or "
            "`cytopert config set providers.<provider>.apiKey ...`.",
        )
    if not cfg.get_api_key():
        return _row(
            "WARN",
            "provider",
            f"Provider {provider!r} selected but the API key is empty.",
        )
    return _row("PASS", "provider", f"{provider} | model={cfg.agents.defaults.model}")


def _check_workspace_writable() -> tuple[str, str, str]:
    from cytopert.config.loader import load_config

    cfg = load_config()
    ws = cfg.workspace_path
    try:
        ws.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=ws, prefix="doctor_", delete=True):
            pass
    except OSError as exc:
        return _row("FAIL", "workspace", f"{ws} not writable: {exc}")
    return _row("PASS", "workspace", str(ws))


def _check_state_db_fts5() -> tuple[str, str, str]:
    from cytopert.persistence.evidence_db import EvidenceDB
    from cytopert.utils.helpers import get_state_db_path

    db_path = get_state_db_path()
    try:
        db = EvidenceDB(db_path)
    except sqlite3.OperationalError as exc:
        if "fts5" in str(exc).lower():
            return _row(
                "FAIL",
                "state.db FTS5",
                "SQLite was built without FTS5; install a python with FTS5 enabled.",
            )
        return _row("FAIL", "state.db", f"open failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _row("FAIL", "state.db", f"open failed: {exc}")

    # Probe an insert / FTS round-trip.
    from cytopert.data.models import EvidenceEntry, EvidenceType

    probe_id = "doctor_probe_xyz"
    try:
        db.add(
            EvidenceEntry(
                id=probe_id,
                type=EvidenceType.DATA,
                summary="cytopert doctor probe row",
                tool_name="doctor_probe",
            ),
            session_id="doctor",
        )
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "DELETE FROM evidence_entries WHERE id = ?", (probe_id,)
            )
            conn.commit()
    except Exception as exc:  # noqa: BLE001
        return _row("FAIL", "state.db FTS5", f"FTS round-trip failed: {exc}")
    return _row("PASS", "state.db", f"opens; FTS5 trigger fires; path={db_path}")


def _check_bundled_skills() -> tuple[str, str, str]:
    from cytopert.skills.manager import SkillsManager
    from cytopert.utils.helpers import get_skills_dir

    skills_dir = get_skills_dir()
    mgr = SkillsManager(skills_dir)
    mgr.install_bundled()  # idempotent after first call
    listed = mgr.list()
    if not listed:
        return _row(
            "WARN",
            "bundled skills",
            f"No SKILL.md found under {skills_dir}; "
            "delete .bundled_manifest and re-run to retry the install.",
        )
    return _row(
        "PASS", "bundled skills", f"{len(listed)} installed at {skills_dir}"
    )


def _check_plugins() -> tuple[str, str, str]:
    from cytopert.plugins.manager import PluginManager

    try:
        infos = PluginManager(project_dir=Path.cwd()).discover()
    except Exception as exc:  # noqa: BLE001
        return _row("FAIL", "plugins", f"discovery raised: {exc}")
    return _row(
        "PASS",
        "plugins",
        f"{len(infos)} plugin(s) discovered (user / project / entry-points)",
    )


def _check_scanpy() -> tuple[str, str, str]:
    # Catch BaseException because heavy scientific imports can raise
    # numba RuntimeError ("no locator available for file ...") on
    # restricted-permission conda installs. Doctor must never crash.
    try:
        import scanpy
    except ImportError as exc:
        return _row("WARN", "scanpy import", f"unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _row(
            "WARN",
            "scanpy import",
            f"raised {type(exc).__name__}: {str(exc)[:120]}",
        )
    return _row("PASS", "scanpy import", f"version {getattr(scanpy, '__version__', '?')}")


def _check_decoupler_op() -> tuple[str, str, str]:
    try:
        import decoupler
    except ImportError as exc:
        return _row("WARN", "decoupler import", f"unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001
        # numba caching errors during decoupler import surface here.
        return _row(
            "WARN",
            "decoupler import",
            f"raised {type(exc).__name__}: {str(exc)[:120]}",
        )
    op = getattr(decoupler, "op", None)
    if op is None:
        return _row(
            "FAIL",
            "decoupler.op",
            f"installed decoupler v{getattr(decoupler, '__version__', '?')} "
            "lacks dc.op; pip install 'decoupler>=2.0'.",
        )
    return _row(
        "PASS",
        "decoupler.op",
        f"version {getattr(decoupler, '__version__', '?')} (dc.op present)",
    )


def _check_ping() -> tuple[str, str, str]:
    """Optional: 1-token LLM round-trip (only when --ping)."""
    from cytopert.config.loader import load_config

    cfg = load_config()
    if not cfg.get_api_key():
        return _row("WARN", "ping", "No API key configured; cannot ping.")
    try:
        from cytopert.providers.litellm_provider import LiteLLMProvider

        provider = LiteLLMProvider(
            api_key=cfg.get_api_key(),
            api_base=cfg.get_api_base(),
            default_model=cfg.agents.defaults.model,
            provider_type=cfg.get_provider_type(),
        )

        async def _go() -> str:
            response = await provider.chat(
                messages=[{"role": "user", "content": "Reply with the single word 'ok'."}],
                tools=None,
                model=cfg.agents.defaults.model,
                max_tokens=4,
                temperature=0.0,
            )
            return response.content or ""

        text = asyncio.run(_go())
    except Exception as exc:  # noqa: BLE001
        return _row("FAIL", "ping", f"call raised: {exc}")
    return _row("PASS", "ping", f"reply: {text[:40]!r}")


def run_doctor(*, ping: bool = False, console: Console | None = None) -> int:
    """Run all health checks and print a table. Returns 0 / 1 exit code."""
    console = console or Console()
    rows: list[tuple[str, str, str]] = []

    rows.append(_check_profile())
    rows.append(_check_config())
    rows.append(_check_provider())
    rows.append(_check_workspace_writable())
    rows.append(_check_state_db_fts5())
    rows.append(_check_bundled_skills())
    rows.append(_check_plugins())
    rows.append(_check_scanpy())
    rows.append(_check_decoupler_op())
    if ping:
        rows.append(_check_ping())

    table = Table(title="CytoPert doctor", show_lines=False)
    table.add_column("Status", style="bold")
    table.add_column("Check")
    table.add_column("Detail")
    for level, name, detail in rows:
        if level == "PASS":
            colour = "green"
        elif level == "WARN":
            colour = "yellow"
        else:
            colour = "red"
        table.add_row(f"[{colour}]{level}[/{colour}]", name, detail)
    console.print(table)

    fail_count = sum(1 for level, _, _ in rows if level == "FAIL")
    return 0 if fail_count == 0 else 1


__all__ = ["run_doctor"]

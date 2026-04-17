"""Job scheduler core: parsing, persistence, and tick loop."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


_INTERVAL_RE = re.compile(r"^\s*every\s+(\d+)\s*([smhd])\s*$", re.IGNORECASE)
_ALIASES: dict[str, timedelta] = {
    "hourly": timedelta(hours=1),
    "daily": timedelta(days=1),
    "minutely": timedelta(minutes=1),
}


def parse_schedule(expr: str) -> timedelta:
    """Parse a schedule string into a fixed timedelta interval.

    Supported grammar (case-insensitive):

    * ``every Ns`` / ``every Nm`` / ``every Nh`` / ``every Nd``
    * ``minutely`` / ``hourly`` / ``daily``

    Raises ``ValueError`` on anything else. The minimum granularity is
    one second; we deliberately reject negative or zero intervals so
    that ``run_due_jobs`` cannot enter a tight infinite loop.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("schedule must be a non-empty string")
    norm = expr.strip().lower()
    if norm in _ALIASES:
        return _ALIASES[norm]
    m = _INTERVAL_RE.match(norm)
    if not m:
        raise ValueError(
            f"unrecognised schedule: {expr!r}; "
            "use 'every Nm/Nh/Nd' or 'hourly' / 'daily'"
        )
    n = int(m.group(1))
    unit = m.group(2)
    if n <= 0:
        raise ValueError("schedule interval must be > 0")
    seconds = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return timedelta(seconds=n * seconds)


def next_run_after(reference: datetime, schedule: str) -> datetime:
    """Return the next run time strictly greater than ``reference``."""
    delta = parse_schedule(schedule)
    return reference + delta


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.warning("Could not parse cron timestamp %r; skipping field", value)
        return None


@dataclass
class Job:
    """A recurring job persisted to ``jobs.json``.

    Exactly one of ``message`` / ``scenario`` is set. ``scenario`` is
    the registered workflow scenario name (see
    ``cytopert.workflow.scenarios``); ``message`` is forwarded to
    ``AgentLoop.process_direct`` as if the user had typed it.
    """

    id: str
    schedule: str
    message: str | None = None
    scenario: str | None = None
    feedback: str | None = None
    enabled: bool = True
    last_run: str | None = None
    next_run: str | None = None
    last_status: str | None = None
    last_error: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def make(
        cls,
        *,
        schedule: str,
        message: str | None = None,
        scenario: str | None = None,
        feedback: str | None = None,
        job_id: str | None = None,
    ) -> "Job":
        """Build a freshly-scheduled Job with validation."""
        if (message is None) == (scenario is None):
            raise ValueError("Job needs exactly one of message / scenario")
        parse_schedule(schedule)  # validate up front
        now = _utcnow()
        return cls(
            id=job_id or f"job_{secrets.token_hex(4)}",
            schedule=schedule,
            message=message,
            scenario=scenario,
            feedback=feedback,
            enabled=True,
            last_run=None,
            next_run=_to_iso(now + parse_schedule(schedule)),
        )

    def is_due(self, *, now: datetime) -> bool:
        if not self.enabled:
            return False
        nxt = _from_iso(self.next_run)
        if nxt is None:
            return True
        return now >= nxt

    def mark_run(self, *, now: datetime, status: str, error: str | None) -> None:
        """Update timestamps and audit history after a run attempt."""
        self.last_run = _to_iso(now)
        self.last_status = status
        self.last_error = error
        self.next_run = _to_iso(now + parse_schedule(self.schedule))
        entry = {
            "ts": _to_iso(now),
            "status": status,
            "error": error or "",
        }
        # Cap history at 20 entries so jobs.json doesn't grow unbounded.
        self.history.append(entry)
        if len(self.history) > 20:
            self.history = self.history[-20:]


class JobStore:
    """File-backed job list with atomic writes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[Job]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("jobs.json unreadable (%s); starting empty", exc)
            return []
        if not isinstance(payload, list):
            logger.warning("jobs.json root must be a list; got %s", type(payload))
            return []
        out: list[Job] = []
        for raw in payload:
            try:
                out.append(Job(**raw))
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping malformed job entry %s: %s", raw, exc)
        return out

    def save(self, jobs: list[Job]) -> None:
        # Atomic write: dump to .tmp + rename so a crash mid-write
        # never leaves jobs.json in a half-written state.
        payload = [asdict(j) for j in jobs]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, job: Job) -> Job:
        jobs = self.load()
        if any(j.id == job.id for j in jobs):
            raise ValueError(f"Job id already exists: {job.id}")
        jobs.append(job)
        self.save(jobs)
        return job

    def remove(self, job_id: str) -> bool:
        jobs = self.load()
        new = [j for j in jobs if j.id != job_id]
        if len(new) == len(jobs):
            return False
        self.save(new)
        return True

    def update(self, job: Job) -> None:
        jobs = self.load()
        for idx, j in enumerate(jobs):
            if j.id == job.id:
                jobs[idx] = job
                self.save(jobs)
                return
        raise KeyError(f"Unknown job id: {job.id}")

    def set_enabled(self, job_id: str, enabled: bool) -> Job:
        jobs = self.load()
        for j in jobs:
            if j.id == job_id:
                j.enabled = enabled
                self.save(jobs)
                return j
        raise KeyError(f"Unknown job id: {job_id}")


# A runner takes a Job and returns (status, error). The default runner
# is bound at CLI invocation time so the scheduler stays free of any
# direct AgentLoop / workflow imports (keeps tests light).
JobRunner = Callable[[Job], "asyncio.Future[tuple[str, str | None]] | tuple[str, str | None]"]


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


async def run_due_jobs(
    store: JobStore,
    runner: JobRunner,
    *,
    now: datetime | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> list[Job]:
    """Run every due job and persist updated metadata. Returns ran jobs."""
    when = now or _utcnow()
    jobs = store.load()
    ran: list[Job] = []
    for job in jobs:
        if not job.is_due(now=when):
            continue
        if on_progress:
            on_progress(f"running {job.id} ({job.schedule}) ...")
        try:
            result = await _maybe_await(runner(job))
            if not isinstance(result, tuple) or len(result) != 2:
                status, err = "ok", None
            else:
                status, err = result
        except Exception as exc:  # noqa: BLE001
            logger.warning("Job %s failed: %s", job.id, exc)
            status, err = "error", f"{type(exc).__name__}: {exc}"
        job.mark_run(now=_utcnow(), status=status, error=err)
        ran.append(job)
    if ran:
        store.save(jobs)
    return ran


async def run_daemon(
    store: JobStore,
    runner: JobRunner,
    *,
    interval_seconds: int = 60,
    stop_event: asyncio.Event | None = None,
    on_tick: Callable[[list[Job]], None] | None = None,
) -> None:
    """Loop forever (until ``stop_event``) calling ``run_due_jobs``."""
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        ran = await run_due_jobs(store, runner)
        if on_tick is not None:
            on_tick(ran)
        # Sleep in small chunks so a stop_event can interrupt promptly.
        slept = 0
        while slept < interval_seconds:
            if stop_event is not None and stop_event.is_set():
                return
            await asyncio.sleep(min(2, interval_seconds - slept))
            slept += 2


def get_default_jobs_path() -> Path:
    """Return ``<profile>/jobs.json`` honouring profile isolation."""
    from cytopert.utils.helpers import get_data_path

    return get_data_path() / "jobs.json"


def synchronous_runner_for_message(message: str, *, scenario: str | None) -> str:
    """Pretty banner for ``cytopert cron list`` to display in the table."""
    if scenario:
        return f"scenario={scenario}"
    return f"message={message[:60]}{'...' if len(message) > 60 else ''}"


__all__ = [
    "Job",
    "JobStore",
    "_to_iso",
    "_utcnow",
    "get_default_jobs_path",
    "next_run_after",
    "parse_schedule",
    "run_daemon",
    "run_due_jobs",
    "synchronous_runner_for_message",
]


# A tiny convenience used by the CLI / tests when they want a runner
# that simply invokes AgentLoop on an already-built loop instance.
def make_agent_runner(
    agent_loop: Any,
    *,
    config: Any | None = None,
) -> JobRunner:
    """Return a runner that dispatches a Job to ``agent_loop``.

    Returns ``("ok", None)`` on success and ``("error", "<repr>")`` on
    exception. Scenario jobs build a fresh ``Pipeline`` from the
    workflow scenario registry and run it once; message jobs go
    through ``AgentLoop.process_direct``.

    The ``config`` argument is required only when a job names a
    ``scenario``; the Pipeline needs it to read scenario-specific
    settings from ``Config.workflow``.
    """

    def _runner(job: Job) -> "asyncio.Future[tuple[str, str | None]]":
        async def _go() -> tuple[str, str | None]:
            try:
                if job.scenario:
                    from cytopert.workflow.pipeline import (
                        StageContext,
                        get_scenario,
                        get_scenario_config,
                    )

                    pipeline = get_scenario(job.scenario)
                    if pipeline is None:
                        return (
                            "error",
                            f"unknown scenario: {job.scenario!r}",
                        )
                    if config is None:
                        return (
                            "error",
                            "scenario job requires a Config "
                            "(pass `config=` to make_agent_runner).",
                        )
                    data_cfg = get_scenario_config(config, job.scenario)
                    ctx = StageContext(
                        config=config,
                        research_question=job.message or "",
                        data_config=data_cfg or {},
                        feedback=job.feedback,
                        session_key=f"cron:{job.id}",
                    )
                    await pipeline.run(agent_loop, ctx)
                else:
                    await agent_loop.process_direct(
                        job.message or "",
                        session_key=f"cron:{job.id}",
                        user_feedback=job.feedback,
                    )
            except Exception as exc:  # noqa: BLE001
                return ("error", f"{type(exc).__name__}: {exc}")
            return ("ok", None)

        return _go()

    return _runner

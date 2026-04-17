"""Stage 13a tests: cron scheduler core + CLI surface."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cytopert.cli.commands import app
from cytopert.scheduler.cron import (
    Job,
    JobStore,
    next_run_after,
    parse_schedule,
    run_due_jobs,
)


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path, monkeypatch):
    """Pin profile root + drop CYTOPERT_HOME so tests are hermetic."""
    fake_root = tmp_path / "cytopert_root"
    fake_root.mkdir()
    monkeypatch.delenv("CYTOPERT_HOME", raising=False)
    import cytopert.utils.helpers as hh

    monkeypatch.setattr(hh, "CYTOPERT_ROOT_DIR", fake_root)
    yield


def _runner() -> CliRunner:
    return CliRunner(mix_stderr=False)


def test_parse_schedule_accepts_intervals_and_aliases() -> None:
    assert parse_schedule("every 30s") == timedelta(seconds=30)
    assert parse_schedule("every 5m") == timedelta(minutes=5)
    assert parse_schedule("every 2h") == timedelta(hours=2)
    assert parse_schedule("every 1d") == timedelta(days=1)
    assert parse_schedule("hourly") == timedelta(hours=1)
    assert parse_schedule("daily") == timedelta(days=1)


def test_parse_schedule_rejects_garbage() -> None:
    for bad in ["", "every", "every 0m", "0", "weekly", "every -1m"]:
        with pytest.raises(ValueError):
            parse_schedule(bad)


def test_next_run_after_uses_interval() -> None:
    base = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    assert next_run_after(base, "every 15m") == base + timedelta(minutes=15)


def test_job_store_round_trip(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.json")
    j = Job.make(schedule="every 1h", message="hello")
    store.add(j)
    loaded = store.load()
    assert len(loaded) == 1
    assert loaded[0].id == j.id
    assert loaded[0].schedule == "every 1h"
    assert loaded[0].message == "hello"
    assert store.remove(j.id) is True
    assert store.load() == []


def test_job_store_atomic_write_survives_corruption(tmp_path: Path) -> None:
    """A corrupt jobs.json should yield an empty list rather than crash."""
    p = tmp_path / "jobs.json"
    p.write_text("not-json", encoding="utf-8")
    store = JobStore(p)
    assert store.load() == []


def test_job_make_rejects_double_or_missing_target() -> None:
    with pytest.raises(ValueError):
        Job.make(schedule="every 1h")
    with pytest.raises(ValueError):
        Job.make(schedule="every 1h", message="x", scenario="y")


@pytest.mark.asyncio
async def test_run_due_jobs_only_runs_due(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.json")
    due_job = Job.make(schedule="every 1h", message="run-me")
    # Force it to be due by backdating next_run.
    due_job.next_run = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    far_job = Job.make(schedule="every 1d", message="not-yet")
    far_job.next_run = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    store.save([due_job, far_job])

    calls: list[str] = []

    def runner(job: Job):
        calls.append(job.id)
        return ("ok", None)

    ran = await run_due_jobs(store, runner)
    assert [j.id for j in ran] == [due_job.id]
    assert calls == [due_job.id]
    # next_run got pushed forward.
    refreshed = {j.id: j for j in store.load()}
    assert refreshed[due_job.id].last_status == "ok"
    assert refreshed[due_job.id].next_run > due_job.next_run


@pytest.mark.asyncio
async def test_run_due_jobs_records_runner_exception(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.json")
    j = Job.make(schedule="every 1h", message="boom")
    j.next_run = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    store.save([j])

    def boom(_job: Job):
        raise RuntimeError("test failure")

    ran = await run_due_jobs(store, boom)
    assert len(ran) == 1
    assert ran[0].last_status == "error"
    assert "RuntimeError: test failure" in (ran[0].last_error or "")


def test_cli_lifecycle_add_list_remove() -> None:
    runner = _runner()
    add = runner.invoke(
        app,
        ["cron", "add", "every 30m", "--message", "summarise data", "--id", "myjob"],
    )
    assert add.exit_code == 0, add.stdout + add.stderr
    listed = runner.invoke(app, ["cron", "list"])
    assert listed.exit_code == 0
    assert "myjob" in listed.stdout
    assert "every 30m" in listed.stdout
    rm = runner.invoke(app, ["cron", "remove", "myjob"])
    assert rm.exit_code == 0
    listed2 = runner.invoke(app, ["cron", "list"])
    assert "No jobs" in listed2.stdout


def test_cli_add_validates_schedule_and_target() -> None:
    runner = _runner()
    bad_schedule = runner.invoke(
        app, ["cron", "add", "every never", "--message", "x"]
    )
    assert bad_schedule.exit_code != 0
    no_target = runner.invoke(app, ["cron", "add", "every 1h"])
    assert no_target.exit_code != 0


def test_cli_enable_disable_round_trip() -> None:
    runner = _runner()
    runner.invoke(
        app, ["cron", "add", "every 1h", "--message", "x", "--id", "tog"]
    )
    assert runner.invoke(app, ["cron", "disable", "tog"]).exit_code == 0
    listed = runner.invoke(app, ["cron", "list"])
    assert "off" in listed.stdout.lower()
    assert runner.invoke(app, ["cron", "enable", "tog"]).exit_code == 0


def test_cli_tick_dry_run_lists_due_jobs() -> None:
    runner = _runner()
    runner.invoke(
        app, ["cron", "add", "every 1h", "--message", "go", "--id", "drytest"]
    )
    # Backdate next_run via the store so the job is "due" without
    # requiring us to wait an hour.
    from cytopert.scheduler.cron import JobStore, get_default_jobs_path

    store = JobStore(get_default_jobs_path())
    jobs = store.load()
    jobs[0].next_run = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    store.save(jobs)

    dry = runner.invoke(app, ["cron", "tick", "--dry-run"])
    assert dry.exit_code == 0
    assert "drytest" in dry.stdout


def test_jobs_json_is_isolated_per_profile(tmp_path: Path, monkeypatch) -> None:
    """Switching profile via -p picks up a different jobs.json."""
    runner = _runner()
    assert runner.invoke(app, ["profile", "new", "studyA"]).exit_code == 0
    assert runner.invoke(
        app,
        ["-p", "studyA", "cron", "add", "every 1h", "--message", "a", "--id", "ja"],
    ).exit_code == 0
    # studyB should not see job from studyA
    assert runner.invoke(app, ["profile", "new", "studyB"]).exit_code == 0
    listed_b = runner.invoke(app, ["-p", "studyB", "cron", "list"])
    assert "ja" not in listed_b.stdout

    import cytopert.utils.helpers as hh

    a_path = (
        hh.CYTOPERT_ROOT_DIR / hh.PROFILES_SUBDIR / "studyA" / "jobs.json"
    )
    assert a_path.exists()
    payload = json.loads(a_path.read_text(encoding="utf-8"))
    assert payload[0]["id"] == "ja"


@pytest.mark.asyncio
async def test_run_daemon_honours_stop_event(tmp_path: Path) -> None:
    """The daemon loop returns promptly when stop_event is set."""
    from cytopert.scheduler.cron import run_daemon

    store = JobStore(tmp_path / "jobs.json")
    stop = asyncio.Event()
    stop.set()  # already set so the loop exits on first iteration

    await asyncio.wait_for(
        run_daemon(store, lambda _j: ("ok", None), interval_seconds=5, stop_event=stop),
        timeout=2.0,
    )

"""Lightweight job scheduler for CytoPert (stage 13).

The scheduler stores a list of recurring jobs in
``<profile>/jobs.json`` and offers two execution modes:

* ``cytopert cron tick``  -- runs every due job exactly once then exits.
  Suitable for an external cron / launchd / systemd timer that wakes the
  process every minute.
* ``cytopert cron daemon`` -- runs ``tick`` in a sleep loop. Useful when
  the user wants a self-contained "always-on" scheduler without
  depending on the host's cron.

The supported schedule grammar is intentionally tiny so the scheduler
has zero extra runtime dependencies:

* ``every Ns`` / ``Nm`` / ``Nh`` / ``Nd`` -- fixed interval
* ``hourly``                              -- every 60 minutes
* ``daily``                               -- every 24 hours

Each job carries a free-form ``message`` (or ``scenario`` name) that
``cron tick`` forwards to ``AgentLoop.process_direct`` /
``run_scenario`` so the same trigger that powers ``cytopert agent`` /
``cytopert run-workflow`` also powers cron.
"""

from cytopert.scheduler.cron import (
    Job,
    JobStore,
    next_run_after,
    parse_schedule,
    run_due_jobs,
)

__all__ = [
    "Job",
    "JobStore",
    "next_run_after",
    "parse_schedule",
    "run_due_jobs",
]

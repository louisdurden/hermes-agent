"""Standalone ownership and restart continuity, using real isolated cron stores."""
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _wait_for(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError("Timed out waiting for isolated scheduler evidence")


def _child(code, home):
    child_env = dict(os.environ, HERMES_HOME=str(home))
    child_env.pop("HERMES_PROFILE", None)
    return subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=child_env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _stop(process):
    if process.poll() is None:
        process.terminate()
    process.wait(timeout=30)


def test_standalone_provider_disables_embedded_and_misfire_dispatch(monkeypatch):
    from cron.scheduler_provider import (
        StandaloneCronScheduler, fire_overdue_jobs, resolve_cron_scheduler,
        scheduler_for_profile_mode,
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"cron": {"provider": "standalone"}})
    provider = resolve_cron_scheduler()
    assert isinstance(provider, StandaloneCronScheduler)
    assert scheduler_for_profile_mode(provider, multiplex_profiles=True) is provider
    with patch("cron.scheduler.tick") as tick:
        provider.start(threading.Event(), profile_homes=lambda: [], can_dispatch=lambda: True)
        assert fire_overdue_jobs(provider) == 0
    tick.assert_not_called()


@pytest.mark.platforms("posix")
def test_daemon_lease_fences_other_process_ticks_and_releases_on_exit(tmp_path):
    from cron.scheduler_service import SchedulerOwnership
    owner = SchedulerOwnership()
    assert owner.claim(tmp_path)
    code = ("import cron.scheduler_tick as t\n"
            "t._tick_admitted = lambda *a, **k: 42\n"
            "raise SystemExit(7 if t.tick() == 0 else 0)")
    try:
        child = _child(code, tmp_path)
        assert child.wait(timeout=15) == 7
    finally:
        owner.close()
    child = _child(code, tmp_path)
    assert child.wait(timeout=15) == 0


@pytest.mark.platforms("posix")
def test_no_agent_job_survives_embedded_ticker_process_restart(tmp_path, monkeypatch):
    from cron.jobs import create_job, get_job, remove_job
    from cron.jobs import use_cron_store

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("cron:\n  provider: standalone\n", encoding="utf-8")
    scripts = home / "scripts"
    scripts.mkdir()
    (scripts / "probe.py").write_text(
        "import time\nfrom pathlib import Path\n"
        "p=Path(__file__).parent.parent\n"
        "(p/'started').write_text('started')\ntime.sleep(5)\n"
        "with (p/'finished').open('a') as f: f.write('once\\n')\nprint('probe-ok')\n",
        encoding="utf-8",
    )
    with use_cron_store(home):
        job = create_job(None, (datetime.now(timezone.utc) + timedelta(seconds=2)).isoformat(),
                         script="probe.py", no_agent=True, deliver="local")
    # These processes run the actual embedded ticker entry point, without adapters or network.
    embedded_code = "import threading\nfrom cron.scheduler_provider import InProcessCronScheduler\nInProcessCronScheduler().start(threading.Event(), interval=.1)"
    daemon = _child("from cron.daemon import run_daemon\nraise SystemExit(run_daemon(interval=.1, multiplex=False))", home)
    embedded = None
    duplicate = None
    try:
        _wait_for(lambda: (home / "started").exists())
        duplicate = _child("from cron.daemon import run_daemon\nraise SystemExit(run_daemon(interval=.1, multiplex=False))", home)
        assert duplicate.wait(timeout=15) == 1
        embedded = _child(embedded_code, home)
        _wait_for(lambda: (home / "cron/ticker_heartbeat").exists())
        _stop(embedded)
        embedded = _child(embedded_code, home)
        _wait_for(lambda: (home / "finished").exists())
        assert daemon.poll() is None
        assert (home / "finished").read_text(encoding="utf-8") == "once\n"
        def succeeded():
            with use_cron_store(home):
                return get_job(job["id"]).get("last_status") == "ok"
        _wait_for(succeeded)
    finally:
        for process in (embedded, duplicate, daemon):
            if process is not None:
                _stop(process)
        with use_cron_store(home):
            remove_job(job["id"])

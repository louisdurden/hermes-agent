"""Real worker activity must keep quiet research runs out of stalled."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_progress as progress
from hermes_cli import kanban_tree_budget as budget


@pytest.fixture
def worker(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kbc.connect()
    task_id = kb.create_task(conn, title="Quiet research", assignee="fleetliterature")
    kb.claim_task(conn, task_id)
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    dispatch._set_worker_pid(conn, task_id, proc.pid)
    try:
        yield conn, task_id, proc
    finally:
        proc.communicate(timeout=10)
        conn.close()


@pytest.mark.parametrize("signal", ["tokens", "log"])
def test_live_worker_active_for_six_minutes_without_board_events(worker, signal):
    conn, task_id, _ = worker
    run_id = kb.get_task(conn, task_id).current_run_id
    log = kb.worker_log_path(task_id, board="alfred-fleet")
    log.parent.mkdir(parents=True, exist_ok=True)
    events_before = len(kb.list_events(conn, task_id))
    for elapsed in range(0, 361, 60):
        if signal == "tokens":
            budget.record_usage(conn, task_id, run_id, 100)
        else:
            with log.open("a") as stream:
                stream.write("tool completed\n")
        assert progress.observe_progress(conn, now=1000 + elapsed) == []
    assert kb.get_task(conn, task_id).status == "running"
    assert len(kb.list_events(conn, task_id)) == events_before


def test_live_worker_with_no_signals_stalls_after_ten_minutes(worker):
    conn, task_id, proc = worker
    for elapsed in range(0, 600, 60):
        assert progress.observe_progress(conn, now=1000 + elapsed) == []
    assert progress.observe_progress(conn, now=1600) == [task_id]
    assert proc.poll() is None  # Stall supervision does not terminate a live process.
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET ended_at = 1 WHERE task_id = ?", (task_id,))
    assert dispatch.reap_terminal_workers(
        conn, signal_fn=lambda *args: pytest.fail("stall must not signal the live worker"),
    ) == []
    assert proc.poll() is None


def test_dead_worker_stalls_before_next_heartbeat(worker):
    conn, task_id, proc = worker
    assert progress.observe_progress(conn, now=1000) == []
    proc.communicate(timeout=10)
    assert progress.observe_progress(conn, now=1001) == [task_id]
    assert "died" in kb.list_comments(conn, task_id)[-1].body


def test_heartbeat_setting_is_loaded_from_profile_config(worker, tmp_path):
    conn, task_id, _ = worker
    (tmp_path / "config.yaml").write_text("kanban:\n  stall_heartbeat_seconds: 420\n")
    assert progress.observe_progress(conn, now=1000) == []
    assert progress.observe_progress(conn, now=1300) == []
    assert progress.observe_progress(conn, now=1420) == []
    assert progress.observe_progress(conn, now=1600) == []
    assert progress.observe_progress(conn, now=1840) == [task_id]


def test_worker_activity_arriving_just_before_park_wins(worker, monkeypatch):
    conn, task_id, _ = worker
    run_id = kb.get_task(conn, task_id).current_run_id
    budget.record_usage(conn, task_id, run_id, 1)
    progress.observe_progress(conn, now=1000)
    progress.observe_progress(conn, now=1300)
    original = progress._park_stalled

    def park_after_usage(conn, task, now, expected_signature=None):
        budget.record_usage(conn, task_id, run_id, 1)
        return original(conn, task, now, expected_signature)

    monkeypatch.setattr(progress, "_park_stalled", park_after_usage)
    assert progress.observe_progress(conn, now=1600) == []
    assert kb.get_task(conn, task_id).status == "running"


def test_stalled_live_worker_remains_subject_to_time_budget(worker):
    conn, task_id, proc = worker
    run_id = kb.get_task(conn, task_id).current_run_id
    budget.record_usage(conn, task_id, run_id, 1)
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET started_at = 1000 WHERE id = ?", (run_id,))
    progress.observe_progress(conn, now=1000)
    progress.observe_progress(conn, now=1300)
    assert progress.observe_progress(conn, now=1600) == [task_id]
    with kb.write_txn(conn):
        conn.execute("UPDATE task_runs SET ended_at = 1600 WHERE id = ?", (run_id,))
        conn.execute("UPDATE kanban_tree_budgets SET budget_seconds = 700 WHERE root_id = ?", (task_id,))
    assert budget.consumption(conn, task_id, now=1800)["seconds"] == 800
    signals = []
    assert budget.enforce(conn, now=1800, signal_fn=lambda *args: signals.append(args)) == [task_id]
    assert kb.get_task(conn, task_id).block_kind == "budget_exceeded"
    assert signals and proc.poll() is None

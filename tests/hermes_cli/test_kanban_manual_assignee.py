"""Claude is a pull-only lane, even if a matching worker profile appears."""
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_progress as progress


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn


def test_manual_owner_without_profile_is_visible_and_assignable(board):
    record = next(a for a in kb.known_assignees(board) if a["name"] == "claude")
    assert not record["on_disk"]
    tid = kb.create_task(board, title="Manual", assignee="Claude")
    assert kb.get_task(board, tid).assignee == "claude"
    assert kb.assign_task(board, tid, "claude")
    assert [t.id for t in kb.list_tasks(board, assignee="claude")] == [tid]
    with pytest.raises(ValueError, match="Unknown assignee"):
        kb.assign_task(board, tid, "missing-worker")


@pytest.mark.parametrize("profile_predicate", [None, lambda name: True])
@pytest.mark.parametrize("status", ["ready", "review"])
def test_two_dispatch_cycles_never_claim_manual_owner(board, monkeypatch, profile_predicate, status):
    tid = kb.create_task(board, title="Interactive skill", assignee="claude")
    if status == "review":
        assert kb.request_review(board, tid, summary="Review manually")
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: profile_predicate)
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "ok")
    monkeypatch.setattr(dispatch, "count_running_tasks_other_boards", lambda board: 0)
    for _ in range(2):
        result = dispatch.dispatch_once(board, spawn_fn=lambda *a, **kw: pytest.fail("manual lane spawned"),
                                       max_spawn=1, max_in_progress=3, reconcile_orphans=False)
        assert result.spawned == []
        task = kb.get_task(board, tid)
        assert task.status == status and task.worker_pid is None and task.claim_lock is None
    assert not dispatch._has_spawnable(board, status)
    assert dispatch._resolve_default_assignee("claude") is None
    rows = board.execute("SELECT * FROM tasks WHERE id=?", (tid,)).fetchall()
    assert not dispatch._any_spawnable_review(board, rows)


def test_manual_claim_is_not_worker_stall_and_lease_can_expire(board, monkeypatch):
    tid = kb.create_task(board, title="Manual", assignee="claude")
    claimed = kb.claim_task(board, tid, ttl_seconds=3600)
    assert claimed.worker_pid is None
    for now in (1000, 1300, 1600, 1900):
        assert progress.observe_progress(board, now=now) == []
    monkeypatch.setattr(dispatch.time, "time", lambda: claimed.started_at + 7200)
    assert dispatch.detect_stale_running(board, stale_timeout_seconds=1) == []
    assert kb.get_task(board, tid).status == "running"
    kb.release_stale_claims(board)
    assert kb.get_task(board, tid).status == "ready"


@pytest.mark.parametrize("status", ["ready", "review"])
def test_reassignment_between_dispatch_snapshot_and_claim_never_spawns(board, monkeypatch, status):
    tid = kb.create_task(board, title="Reassigned", assignee="default")
    if status == "review":
        assert kb.request_review(board, tid, summary="Manual review")
    real_claim = kb.claim_task if status == "ready" else kb.claim_review_task

    def racing_claim(conn, task_id, **kwargs):
        assert kb.assign_task(conn, task_id, "claude")
        return real_claim(conn, task_id, **kwargs)

    monkeypatch.setattr(kb, "claim_task" if status == "ready" else "claim_review_task", racing_claim)
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda name: True)
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "ok")
    monkeypatch.setattr(dispatch, "count_running_tasks_other_boards", lambda board: 0)
    dispatch.dispatch_once(board, spawn_fn=lambda *a, **kw: pytest.fail("reassigned manual task spawned"),
                           max_spawn=1, max_in_progress=3, reconcile_orphans=False)
    task = kb.get_task(board, tid)
    assert task.assignee == "claude" and task.status == status and task.claim_lock is None


def test_claim_expected_owner_refuses_foreign_task_without_mutation(board):
    tid = kb.create_task(board, title="Foreign", assignee="default")
    before = kb.get_task(board, tid)
    assert kb.claim_task(board, tid, expected_assignee="claude") is None
    assert kb.get_task(board, tid) == before


def test_old_attempt_cannot_complete_new_manual_claim(board):
    tid = kb.create_task(board, title="Manual attempt", assignee="claude")
    first = kb.claim_task(board, tid)
    assert kb.reclaim_task(board, tid, reason="new manual session")
    second = kb.claim_task(board, tid)
    assert second.current_run_id != first.current_run_id
    assert not kb.complete_task(board, tid, result="old result", expected_run_id=first.current_run_id)
    assert kb.get_task(board, tid).current_run_id == second.current_run_id
    assert kb.get_task(board, tid).status == "running"



def test_manual_mission_uses_coordinator_transport_preserving_author(board, monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from hermes_cli import kanban_mission_notebook as notebook

    tid = kb.create_task(board, title="Manual mission", assignee="claude")
    monkeypatch.setattr(notebook, "mapping", lambda conn: {
        "profiles": {"default": "fleet/missions/alfred"}})
    assert notebook.mission_for(board, kb.get_task(board, tid)) == "fleet/missions/alfred"
    captured = []
    def run(*args, **kwargs):
        captured.append(json.loads(kwargs["input"]))
        return SimpleNamespace(returncode=0, stdout='{"ok":true}')
    monkeypatch.setattr(notebook.subprocess, "run", run)
    notebook.exchange(tmp_path, "claude", "complete", event={"profile": "claude"})
    assert captured[0]["profile"] == "default"
    assert captured[0]["event"]["profile"] == "claude"

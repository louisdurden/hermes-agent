"""Fleet progress, Alfred-only diagnostics and bounded dispatcher contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_progress as progress


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kbc.connect()
    yield conn
    conn.close()


def running(conn, **kwargs):
    task_id = kb.create_task(conn, title="Observe research", assignee="fleetliterature", **kwargs)
    if kb.get_task(conn, task_id).status != "running":
        kb.claim_task(conn, task_id)
    return task_id


def test_two_progress_heartbeats_ignore_worker_keepalives_and_resume_after_restart(board, tmp_path):
    task_id = running(board)
    assert progress.observe_progress(board, now=1000) == []
    dispatch.heartbeat_worker(board, task_id, note="still alive")
    assert progress.observe_progress(board, now=1299) == []
    assert progress.observe_progress(board, now=1300) == []
    # New connection simulates a dispatcher restart; snapshots stay durable.
    other = kbc.connect()
    try:
        assert progress.observe_progress(other, now=1600) == [task_id]
    finally:
        other.close()
    task = kb.get_task(board, task_id)
    assert task.status == "blocked" and task.block_kind == "stalled"
    assert "stalled" in kb.list_comments(board, task_id)[-1].body
    events = kb.list_events(board, task_id)
    assert any(e.kind == "progress_stalled" for e in events)
    assert not any(e.kind == "blocked" for e in events)
    inbox = tmp_path / "kanban" / "supervisor-inbox" / "alfred-fleet"
    notice = json.loads(next(inbox.glob("stalled-*.json")).read_text())
    assert notice["recipient"] == "alfred" and notice["task_id"] == task_id
    assert progress.observe_progress(board, now=1900) == []
    assert len(list(inbox.glob("stalled-*.json"))) == 1


def test_comments_and_artifact_stat_changes_reset_idle_heartbeat(board, tmp_path):
    artifact = tmp_path / "research.md"
    artifact.write_text("first")
    task_id = running(board, body=f"artifact_path: {artifact}")
    progress.observe_progress(board, now=1000)
    progress.observe_progress(board, now=1300)
    kb.add_comment(board, task_id, "fleetliterature", "Source reviewed")
    assert progress.observe_progress(board, now=1600) == []
    progress.observe_progress(board, now=1900)
    artifact.write_text("a changed result")
    assert progress.observe_progress(board, now=2200) == []
    assert progress.observe_progress(board, now=2500) == []
    assert progress.observe_progress(board, now=2800) == [task_id]


def test_daily_summary_groups_old_blocks_once_and_keeps_new_blocks_out(board, tmp_path):
    now = 400000
    old = running(board)
    recent = running(board)
    kb.block_task(board, old, reason="Need input", kind="needs_input")
    kb.block_task(board, recent, reason="Need input", kind="needs_input")
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET created_at = ? WHERE id = ?", (1, old))
        board.execute("UPDATE task_events SET created_at = ? WHERE task_id = ?", (1, old))
    progress.observe_progress(board, now=now)
    progress.observe_progress(board, now=now + 60)
    inbox = tmp_path / "kanban" / "supervisor-inbox" / "alfred-fleet"
    summaries = list(inbox.glob("daily-*.txt"))
    assert len(summaries) == 1
    assert summaries[0].read_text().count("\n") == 1
    assert "fleetliterature/needs_input=1" in summaries[0].read_text()


def test_fleet_budget_never_exceeds_three_even_with_larger_config(board, monkeypatch):
    for _ in range(2):
        running(board)
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "ok")
    monkeypatch.setattr(dispatch, "count_running_tasks_other_boards", lambda board: 0)
    assert dispatch._tick_spawn_budget(board, dispatch.DispatchResult(), max_spawn=None,
                                       max_in_progress=None, board="alfred-fleet") == (True, 1)
    assert dispatch._tick_spawn_budget(board, dispatch.DispatchResult(), max_spawn=20,
                                       max_in_progress=20, board="alfred-fleet") == (True, 1)
    running(board)
    assert dispatch._tick_spawn_budget(board, dispatch.DispatchResult(), max_spawn=None,
                                       max_in_progress=None, board="alfred-fleet")[0] is False


def test_dry_run_and_other_boards_do_not_observe_progress(board, monkeypatch):
    monkeypatch.setattr(dispatch, "_run_reclaim_phase", lambda *a, **kw: None)
    monkeypatch.setattr(dispatch, "_tick_spawn_budget", lambda *a, **kw: (False, 0))
    monkeypatch.setattr(progress, "observe_progress", lambda *a, **kw: pytest.fail("must not observe"))
    dispatch._dispatch_once_locked(board, board="alfred-fleet", dry_run=True)
    dispatch._dispatch_once_locked(board, board="default")
    assert not board.execute("SELECT 1 FROM sqlite_master WHERE name = 'fleet_progress_snapshots'").fetchone()


def test_dispatch_respects_assignee_profile_limit_and_fleet_cap(board, tmp_path, monkeypatch):
    owners = ["fleetliterature", "fleetliterature", "fleetresearch", "fleetcontent", "fleetops"]
    ids = [kb.create_task(board, title=f"Task {i}", assignee=owner, workspace_kind="dir",
                          workspace_path=str(tmp_path)) for i, owner in enumerate(owners)]
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET status = 'ready', claim_lock = NULL, current_run_id = NULL")
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda name: True)
    monkeypatch.setattr(dispatch, "_run_reclaim_phase", lambda *a, **kw: None)
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "ok")
    monkeypatch.setattr(dispatch, "check_respawn_guard", lambda *a, **kw: None)
    spawned = []
    result = dispatch._dispatch_once_locked(
        board, board="alfred-fleet", max_spawn=20, max_in_progress_per_profile=1,
        spawn_fn=lambda task, workspace: spawned.append((task.id, task.assignee)),
    )
    assert len(result.spawned) == 3
    assert len({owner for _, owner in spawned}) == 3
    assert all(owner == kb.get_task(board, task_id).assignee for task_id, owner in spawned)
    assert result.skipped_per_profile_capped == [(ids[1], "fleetliterature", 1)]
    assert kb.get_task(board, ids[-1]).status == "ready"


def test_progress_arriving_between_observation_and_park_wins(board, monkeypatch):
    task_id = running(board)
    progress.observe_progress(board, now=1000)
    progress.observe_progress(board, now=1300)
    original = progress._park_stalled

    def park_after_comment(conn, task, now, expected_signature=None):
        kb.add_comment(conn, task_id, "fleetliterature", "Artifact verified immediately before parking")
        return original(conn, task, now, expected_signature)

    monkeypatch.setattr(progress, "_park_stalled", park_after_comment)
    assert progress.observe_progress(board, now=1600) == []
    assert kb.get_task(board, task_id).status == "running"


def test_dispatch_does_not_repromote_or_spawn_stalled_card(board, monkeypatch):
    task_id = running(board)
    progress.observe_progress(board, now=1000)
    progress.observe_progress(board, now=1300)
    assert progress.observe_progress(board, now=1600) == [task_id]
    monkeypatch.setattr(dispatch, "_memory_pressure_level", lambda: "ok")
    result = dispatch.dispatch_once(
        board, board="alfred-fleet", reconcile_orphans=False,
        spawn_fn=lambda task, workspace: None,
    )
    assert all(item[0] != task_id and item[1] == "default" for item in result.spawned)
    assert kb.get_task(board, task_id).status == "blocked"


@pytest.mark.parametrize('terminal', ['done', 'archived', 'blocked'])
def test_legacy_alerts_archive_and_only_active_origins_reach_brief(board, terminal):
    origin = kb.create_task(board, title='origin', assignee='fleetcontent')
    alert = kb.create_task(board, title='Alfred: budget_exceeded '+origin,
                           assignee='default', created_by='alfred-tree-policy')
    unrelated = kb.create_task(board, title='unrelated', assignee='fleetcontent')
    with kb.write_txn(board):
        board.execute('UPDATE tasks SET status=? WHERE id=?', (terminal, origin))
        board.execute("UPDATE tasks SET status='blocked' WHERE id=?", (alert,))
    assert progress.reconcile_alerts(board) == [alert]
    assert progress.reconcile_alerts(board) == []
    assert kb.get_task(board, alert).status == 'archived'
    assert kb.list_comments(board, alert)
    assert kb.get_task(board, unrelated).status == 'ready'
    assert (origin in progress.active_alert_summary(board)) == (terminal == 'blocked')

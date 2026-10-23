"""X-2: invalid profiles never enter the queue or release a live claim."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_graph import decompose_triage_task


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "fleetseo"
    profile.mkdir(parents=True)
    (profile / "SOUL.md").write_text("SEO local")
    (home / "profiles" / "ghost").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn


@pytest.mark.parametrize("surface", ["db_create", "db_assign", "reclaim", "specify", "decompose", "tool", "cli_create", "cli_assign"])
def test_unknown_profile_rejected_without_mutation(board, surface):
    tid = kb.create_task(board, title="Existing", assignee="fleetseo", triage=surface in {"specify", "decompose"})
    if surface == "reclaim":
        kb.claim_task(board, tid)
    before = list(board.execute("SELECT * FROM tasks").fetchall())
    events = board.execute("SELECT count(*) FROM task_events").fetchone()[0]
    if surface.startswith("cli_"):
        command = "create Probe --assignee seo-local" if surface == "cli_create" else f"assign {tid} seo-local"
        error = cli.run_slash(command)
    elif surface == "tool":
        from tools import kanban_tools
        error = json.loads(kanban_tools._handle_create({"title": "Probe", "assignee": "seo-local"}))["error"]
    else:
        with pytest.raises(ValueError) as caught:
            actions = {
                "db_create": lambda: kb.create_task(board, title="Probe", assignee="seo-local"),
                "db_assign": lambda: kb.assign_task(board, tid, "seo-local"),
                "reclaim": lambda: kb.reassign_task(board, tid, "seo-local", reclaim_first=True),
                "specify": lambda: kb.specify_triage_task(board, tid, assignee="seo-local"),
                "decompose": lambda: decompose_triage_task(board, tid, root_assignee="default", children=[{"title": "child", "assignee": "seo-local"}]),
            }
            actions[surface]()
        error = str(caught.value)
    assert "seo-local" in error and "Valid profiles:" in error
    assert "fleetseo" in error and "default" in error and "alfred" in error
    assert list(board.execute("SELECT * FROM tasks").fetchall()) == before
    assert board.execute("SELECT count(*) FROM task_events").fetchone()[0] == events


@pytest.mark.parametrize("profile,expected", [("FleetSEO", "fleetseo"), ("default", "default"), ("Alfred", "default"), (None, None)])
def test_existing_profiles_and_self_alias_round_trip(board, profile, expected):
    tid = kb.create_task(board, title="Probe", assignee=profile)
    assert kb.get_task(board, tid).assignee == expected
    assert kb.assign_task(board, tid, "default")
    assert kb.assign_task(board, tid, profile)
    assert kb.get_task(board, tid).assignee == expected


def test_profile_registry_is_live_and_filters_can_query_old_names(board):
    from hermes_cli.profiles import _get_profiles_root
    with pytest.raises(ValueError, match="Valid profiles:"):
        kb.create_task(board, title="Ghost", assignee="ghost")
    assert kb.list_tasks(board, assignee="former-profile") == []
    profile = _get_profiles_root() / "new-worker"
    profile.mkdir()
    (profile / "SOUL.md").write_text("New worker")
    tid = kb.create_task(board, title="New", assignee="new-worker")
    assert kb.get_task(board, tid).assignee == "new-worker"
    (profile / "SOUL.md").unlink()
    with pytest.raises(ValueError, match="Valid profiles:"):
        kb.assign_task(board, tid, "new-worker")


def snapshot(conn):
    return {table: list(conn.execute(f"SELECT * FROM {table}").fetchall())
            for table in ("tasks", "task_events", "task_runs", "task_comments")}


@pytest.mark.parametrize("kind", [None, "listing"])
def test_unknown_reviewer_rejected_before_handoff(board, kind):
    tid = kb.create_task(board, title="Review", assignee="fleetseo",
                         body=f"task_type: {kind}" if kind else None)
    before = snapshot(board)
    with pytest.raises(ValueError, match="Valid profiles:"):
        kb.request_review(board, tid, reviewer="seo-local", summary="Candidate")
    assert snapshot(board) == before


@pytest.mark.parametrize("operation", ["reopen", "changes", "cross_review"])
def test_deleted_implementer_never_restored(board, operation):
    from hermes_cli.profiles import _get_profiles_root
    from hermes_cli import kanban_cross_review as cross
    tid = kb.create_task(board, title="Implement", assignee="fleetseo")
    assert kb.request_review(board, tid, reviewer="default")
    if operation == "changes":
        assert kb.claim_review_task(board, tid) is not None
    if operation == "cross_review":
        cross.ensure_schema(board)
        rid = kb.create_task(board, title="Independent review", assignee="default")
        board.execute("UPDATE tasks SET status='done',result=? WHERE id=?",
                      (json.dumps({"verdict": "FAIL", "proof": "Local review"}), rid))
        original = kb.get_task(board, tid)
        board.execute("INSERT INTO kanban_cross_reviews VALUES (?,?,?,?,?,?,?)",
                      (rid, tid, cross._hash(original.result, original.body), "fleetseo", "default", None, None))
        board.commit()
    (_get_profiles_root() / "fleetseo" / "SOUL.md").unlink()
    before = snapshot(board)
    with pytest.raises(ValueError, match="Valid profiles:"):
        if operation == "reopen":
            kb.reopen_review_task(board, tid)
        elif operation == "changes":
            kb.request_changes(board, tid, reason="Needs work")
        else:
            cross.finish(board, rid)
    assert snapshot(board) == before
    if operation == "cross_review":
        assert board.execute("SELECT verdict FROM kanban_cross_reviews WHERE review_id=?", (rid,)).fetchone()[0] is None


@pytest.mark.parametrize("operation", ["create", "assign"])
def test_cli_process_returns_failure_for_unknown_profile(board, operation):
    tid = kb.create_task(board, title="Existing", assignee="default")
    args = ["create", "Probe", "--assignee", "seo-local"] if operation == "create" else ["assign", tid, "seo-local"]
    before = snapshot(board)
    result = subprocess.run([sys.executable, "-m", "hermes_cli.main", "kanban", *args],
                            env=dict(os.environ), capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "Valid profiles:" in result.stderr and "fleetseo" in result.stderr
    assert snapshot(board) == before


@pytest.mark.parametrize("invalid", ["", " ", 123, "../fleetseo"])
def test_malformed_assignee_still_lists_valid_profiles(board, invalid):
    with pytest.raises(ValueError, match="Valid profiles:.*fleetseo"):
        kb.create_task(board, title="Invalid", assignee=invalid)

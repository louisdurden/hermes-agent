"""Only Alfred's configured default-profile cron jobs consume fleet notices."""
from datetime import datetime, timezone
from pathlib import Path

import pytest

from cron import scheduler_prompt
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_progress as progress


@pytest.fixture
def inbox(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    kb.create_board("alfred-fleet")
    kb.init_db(board="alfred-fleet")
    conn = kbc.connect(board="alfred-fleet")
    task = kb.create_task(conn, title="Stalled research", assignee="fleetliterature", board="alfred-fleet")
    if kb.get_task(conn, task).status != "running":
        kb.claim_task(conn, task)
    progress._ensure_schema(conn)
    progress._park_stalled(conn, conn.execute("SELECT * FROM tasks WHERE id = ?", (task,)).fetchone(), 1000)
    day = datetime.now(timezone.utc).date().isoformat()
    with kb.write_txn(conn):
        conn.execute("INSERT INTO fleet_supervisor_inbox VALUES (?, ?)", (f"daily-{day}", "Alfred: bloqueadas >72 h: fleetliterature/needs_input=2"))
    yield conn, task, home
    conn.close()


def test_internal_notice_is_a_private_alfred_task_not_telegram_cron_context(inbox):
    conn, task, home = inbox
    prompt = scheduler_prompt._build_job_prompt({"id": progress.ALFRED_SUPERVISOR_JOB, "prompt": "Supervisa la flota"})
    assert task not in prompt
    alert = conn.execute("SELECT * FROM tasks WHERE created_by = ?", (progress._SUPERVISOR,)).fetchone()
    assert alert["assignee"] == "default" and task in alert["body"]
    assert conn.execute("SELECT COUNT(*) FROM kanban_notify_subs WHERE task_id = ?", (alert["id"],)).fetchone()[0] == 0
    assert task not in scheduler_prompt._build_job_prompt({"id": "unrelated", "prompt": "Otro trabajo"})
    assert progress.render_supervisor_context(progress.ALFRED_SUPERVISOR_JOB) == ""


def test_daily_brief_gets_only_grouped_line_and_resolved_notice_disappears(inbox):
    conn, task, home = inbox
    prompt = scheduler_prompt._build_job_prompt({"id": progress.ALFRED_BRIEF_JOB, "prompt": "Resume el día"})
    assert "fleetliterature/needs_input=2" in prompt
    assert task not in prompt
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET block_kind = 'needs_input' WHERE id = ?", (task,))
    assert progress.render_supervisor_context(progress.ALFRED_SUPERVISOR_JOB) == ""


def test_same_job_id_in_other_profile_cannot_read_alfred_inbox(inbox, monkeypatch):
    conn, task, home = inbox
    other = home / "profiles" / "fleetliterature"
    other.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(other))
    assert progress.render_supervisor_context(progress.ALFRED_SUPERVISOR_JOB) == ""
    assert progress.render_supervisor_context(progress.ALFRED_BRIEF_JOB) == ""

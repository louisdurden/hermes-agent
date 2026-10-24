"""L4-14: read-before-work, durable completion and CAS/idempotent shared writes."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_mission_notebook as notebook


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "alfred-fleet")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban/boards/alfred-fleet/kanban.db"))
    kb.init_db()
    mapping = kb.kanban_db_path().parent / "mission-notebooks.json"
    mapping.write_text(json.dumps({"version": 1, "profiles": {
        "default": "fleet/missions/alfred", "fleetseo": "fleet/missions/seo"}}))
    with kbc.connect() as conn:
        yield conn


def task(board, body=""):
    return kb.create_task(board, title="fixture", assignee="default", body=body)


def test_start_injects_real_page_and_refuses_unread_page(board, monkeypatch):
    tid = task(board)
    seen = []
    def exchange(*args, **kwargs):
        seen.append((args, kwargs))
        return {"ok": True, "content": "\n".join(f"## {s}\nPlan compartido real" for s in notebook.SECTIONS)}
    monkeypatch.setattr(notebook, "exchange", exchange)
    context = kb.build_worker_context(board, tid)
    assert "Plan compartido real" in context
    assert seen[0][0][1:] == ("default", "read")
    def offline(*args, **kwargs):
        raise RuntimeError("offline")
    monkeypatch.setattr(notebook, "exchange", offline)
    with pytest.raises(RuntimeError):
        kb.build_worker_context(board, tid)


def test_explicit_mission_and_invalid_slug(board):
    tid = task(board, "mission_page: fleet/missions/otra")
    assert notebook.mission_for(board, kb.get_task(board, tid)) == "fleet/missions/otra"
    tid = task(board, "mission_page: people/privado")
    with pytest.raises(ValueError):
        notebook.mission_for(board, kb.get_task(board, tid))


def test_completion_outbox_survives_failure_and_retries_once(board, monkeypatch, tmp_path):
    tid = task(board)
    assert kb.claim_task(board, tid)
    proof = tmp_path / "proof.md"
    proof.write_text("Evidencia sintética durable")
    monkeypatch.setattr(notebook, "exchange", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    assert kb.complete_task(board, tid, result=json.dumps({"artifact_path": str(proof)}))
    assert board.execute("SELECT count(*) FROM task_events WHERE kind='mission_notebook_pending'").fetchone()[0] == 1
    assert not board.execute("SELECT 1 FROM task_events WHERE kind='mission_notebook_synced'").fetchone()
    calls = []
    def online(*args, **kwargs):
        calls.append(kwargs)
        # Remote call is after local commit, and cannot see a half-closed task.
        with kbc.connect() as other:
            assert kb.get_task(other, tid).status == "done"
        return {"ok": True}
    monkeypatch.setattr(notebook, "exchange", online)
    assert notebook.flush_pending(board) == [{"task_id": tid, "ok": True}]
    assert notebook.flush_pending(board) == []
    assert len(calls) == 1


def test_default_board_has_no_remote_effect(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "default")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    kb.init_db()
    monkeypatch.setattr(notebook, "exchange", lambda *a, **k: pytest.fail("remote effect"))
    with kbc.connect() as conn:
        tid = task(conn)
        assert kb.claim_task(conn, tid)
        assert kb.complete_task(conn, tid, result="synthetic proof")


class Session:
    def __init__(self, conflict=False):
        self.content = "\n".join(f"## {s}\n" for s in notebook.SECTIONS)
        self.revision = 0
        self.conflict = conflict
        self.writes = 0

    async def call_tool(self, name, args):
        if name == "get_page":
            data = {"content": self.content, "revision": str(self.revision)}
        else:
            self.writes += 1
            if self.conflict:
                self.conflict = False
                self.content += "\nCambio ajeno conservado\n"
                self.revision += 1
                data = {"error": "revision_conflict"}
            else:
                assert args["expected_revision"] == str(self.revision)
                for edit in args["edits"]:
                    self.content = self.content.replace(edit["old_text"], edit["new_text"])
                self.revision += 1
                data = {"ok": True}
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(data))])


def test_remote_cas_preserves_peer_and_is_idempotent():
    session = Session(conflict=True)
    request = {"slug": "fleet/missions/probe", "operation": "complete", "event": {
        "task_id": "t_probe", "pending_id": 1, "profile": "fleetseo", "completed_at": 123}}
    assert asyncio.run(notebook.execute(session, request))["ok"]
    assert "Cambio ajeno conservado" in session.content
    assert session.content.count("l4-14:t_probe:1") == 2
    assert asyncio.run(notebook.execute(session, request))["ok"]
    assert session.writes == 2


def test_remote_write_without_readback_is_not_success():
    session = Session()
    original = session.call_tool
    async def no_write(name, args):
        if name == "edit_page":
            return SimpleNamespace(content=[SimpleNamespace(type="text", text='{"ok":true}')])
        return await original(name, args)
    session.call_tool = no_write
    request = {"slug": "fleet/missions/probe", "operation": "complete", "event": {
        "task_id": "t_probe", "pending_id": 1, "profile": "fleetseo", "completed_at": 123}}
    with pytest.raises(ValueError, match="not_verified"):
        asyncio.run(notebook.execute(session, request))


def test_failed_old_pending_does_not_starve_new_closures(board, monkeypatch, tmp_path):
    first, second = task(board), task(board)
    with kb.write_txn(board):
        for tid in (first, second):
            notebook.stage_completion(board, kb.get_task(board, tid))
    calls = []
    def exchange(*args, **kwargs):
        tid = kwargs["event"]["task_id"]
        calls.append(tid)
        if tid == first:
            raise RuntimeError("unavailable_old_profile")
        return {"ok": True}
    monkeypatch.setattr(notebook, "exchange", exchange)
    assert notebook.flush_pending(board, limit=1) == [{"task_id": first, "ok": False}]
    assert notebook.flush_pending(board, limit=1) == [{"task_id": second, "ok": True}]
    assert calls == [first, second]


def test_context_reserves_every_section_after_large_history():
    raw = "\n".join(f"## {s}\n" + ("x" * 30000 if s == "Estado" else f"dato de {s}")
                    for s in notebook.SECTIONS)
    rendered = notebook.bounded_content(raw)
    assert len(rendered) < 16000
    for section in notebook.SECTIONS:
        assert f"## {section}" in rendered
    assert "dato de Decisiones" in rendered


def test_missing_profile_credential_never_uses_launch_environment(tmp_path, monkeypatch):
    home = tmp_path / "profiles/fleetseo"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text('mcp_servers:\n  gbrain:\n    url: http://localhost/mcp\n'
                                      '    headers:\n      Authorization: "Bearer ${TEST_GBRAIN_KEY}"\n')
    monkeypatch.setenv("TEST_GBRAIN_KEY", "synthetic-launch-only")
    with pytest.raises(ValueError, match="missing_gbrain_credential"):
        notebook.server_config(tmp_path, "fleetseo")


def test_sensitive_lesson_is_not_staged(board):
    tid = task(board)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET result=? WHERE id=?", (json.dumps({"mission_learning": "test@example.invalid"}), tid))
        notebook.stage_completion(board, kb.get_task(board, tid))
    payload = json.loads(board.execute("SELECT payload FROM task_events WHERE kind='mission_notebook_pending'").fetchone()[0])
    assert payload["learning"] is None

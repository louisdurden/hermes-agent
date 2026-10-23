"""L4-2: missions and effective budgets at the durable creation boundary."""
from pathlib import Path
import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_graph import decompose_triage_task

SPEC = "Objetivo: auditar\nTerminado cuando: tests pasan\nRecibo: /tmp/prueba.json"
ENGLISH = "Goal: audit\nDone when: tests pass\nReceipt: /tmp/proof.json"


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn, home


def configure(home, mode="enforce", budgets=True):
    # Real effective loader; never a production home.
    content = f"kanban:\n  require_spec: {mode}\n"
    if budgets:
        content += "  budget_defaults:\n    budget_tokens: 2000000\n    budget_seconds: 2400\n"
    (home / "config.yaml").write_text(content)


def comments(conn, tid):
    return [row[0] for row in conn.execute("SELECT body FROM task_comments WHERE task_id=?", (tid,))]


@pytest.mark.parametrize("body", [SPEC, ENGLISH])
def test_complete_mission(board, body):
    conn, home = board
    configure(home)
    tid = kb.create_task(conn, title="Audit", body=body)
    assert kb.get_task(conn, tid).body == body
    assert not comments(conn, tid)


@pytest.mark.parametrize("missing", ["Objetivo", "Terminado cuando", "Recibo"])
@pytest.mark.parametrize("mode", ["warn", "enforce"])
def test_each_missing_part_and_transition(board, missing, mode):
    conn, home = board
    configure(home, mode)
    body = "\n".join(line for line in SPEC.splitlines() if not line.startswith(missing + ":"))
    if mode == "enforce":
        with pytest.raises(ValueError, match="falta especificación: " + missing):
            kb.create_task(conn, title="Audit", body=body)
        assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0
    else:
        tid = kb.create_task(conn, title="Audit", body=body)
        assert comments(conn, tid) == ["falta especificación: " + missing]


@pytest.mark.parametrize("missing", ["budget_tokens", "budget_seconds"])
def test_no_effective_budget_rejected(board, missing):
    conn, home = board
    configure(home, budgets=False)
    other = "budget_seconds: 10" if missing == "budget_tokens" else "budget_tokens: 100"
    with pytest.raises(ValueError, match=missing):
        kb.create_task(conn, title="Audit", body=SPEC + "\n" + other)
    assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


def test_explicit_and_typed_defaults(board):
    conn, home = board
    configure(home, budgets=False)
    kb.create_task(conn, title="Explicit", body=SPEC + "\nbudget_tokens: 50\nbudget_seconds: 10")
    (home / "config.yaml").write_text("kanban:\n  require_spec: enforce\n  budget_defaults:\n    research:\n      budget_tokens: 100\n      budget_seconds: 20\n")
    kb.create_task(conn, title="Typed", body=SPEC + "\ntask_type: research")


@pytest.mark.parametrize("decompose", [False, True])
def test_child_inherits_without_losing_own_instructions(board, decompose):
    conn, home = board
    configure(home)
    root = kb.create_task(conn, title="Parent", body=SPEC + "\nbudget_tokens: 100\nbudget_seconds: 10", triage=decompose)
    configure(home, budgets=False)
    own = "Objetivo: revisar solo módulo A"
    if decompose:
        tid = decompose_triage_task(conn, root, root_assignee="default", children=[{"title": "Child", "body": own}])[0]
    else:
        tid = kb.create_task(conn, title="Child", body=own, creator_task_id=root)
    body = kb.get_task(conn, tid).body
    assert body.startswith(own)
    assert "Objetivo: auditar" not in body
    assert "Terminado cuando: tests pasan" in body
    assert "Recibo: /tmp/prueba.json" in body
    assert comments(conn, tid) == [f"especificación heredada de {root}: Terminado cuando, Recibo"]


def test_default_warn_and_empty_sections(board):
    conn, _ = board
    tid = kb.create_task(conn, title="Legacy", body="Goal:\nDone when:\nReceipt:")
    assert comments(conn, tid) == ["falta especificación: Objetivo, Terminado cuando, Recibo, budget_tokens, budget_seconds"]


@pytest.mark.parametrize("surface", ["cli", "tool"])
def test_public_creation_rejects_without_insert(board, surface):
    conn, home = board
    configure(home)
    if surface == "tool":
        from tools.kanban_tools import _handle_create
        error = json.loads(_handle_create({"title": "Probe", "assignee": "default"}))["error"]
    else:
        from hermes_cli.kanban import run_slash
        error = run_slash("create Probe")
    assert "falta especificación:" in error
    assert conn.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0


@pytest.mark.parametrize("decompose", [False, True])
def test_child_with_no_body_inherits_all_sections(board, decompose):
    conn, home = board
    configure(home)
    root = kb.create_task(conn, title="Parent", body=SPEC, triage=decompose)
    if decompose:
        tid = decompose_triage_task(conn, root, root_assignee="default", children=[{"title": "Child"}])[0]
    else:
        tid = kb.create_task(conn, title="Child", creator_task_id=root)
    for line in SPEC.splitlines():
        assert line in kb.get_task(conn, tid).body
    assert comments(conn, tid) == [f"especificación heredada de {root}: Objetivo, Terminado cuando, Recibo"]


def test_decomposition_rejection_rolls_back_entire_graph(board):
    conn, home = board
    configure(home, "warn")
    root = kb.create_task(conn, title="Legacy parent", triage=True)
    configure(home)
    before = {table: list(conn.execute(f"SELECT * FROM {table}"))
              for table in ("tasks", "task_events", "task_comments", "task_links")}
    with pytest.raises(ValueError, match="falta especificación:"):
        decompose_triage_task(conn, root, root_assignee="default", children=[
            {"title": "Complete", "body": SPEC}, {"title": "Incomplete"},
        ])
    for table, rows in before.items():
        assert list(conn.execute(f"SELECT * FROM {table}")) == rows


def test_complete_child_keeps_own_spec(board):
    conn, home = board
    configure(home)
    root = kb.create_task(conn, title="Parent", body=SPEC)
    tid = kb.create_task(conn, title="Child", body=ENGLISH, creator_task_id=root)
    assert kb.get_task(conn, tid).body == ENGLISH
    assert not comments(conn, tid)


def test_metadata_does_not_fill_an_empty_receipt(board):
    conn, home = board
    configure(home)
    body = "Goal: audit\nDone when: pass\nReceipt:\nbudget_tokens: 100"
    with pytest.raises(ValueError, match="falta especificación: Recibo"):
        kb.create_task(conn, title="Empty receipt", body=body)

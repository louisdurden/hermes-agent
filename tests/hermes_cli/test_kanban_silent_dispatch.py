"""D3: exceeded ready cards cannot silently cycle; historical caps are atomic."""
import pytest
from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_tree_budget as budget
from gateway.kanban_dispatch_health import observe, task_causes


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setattr(budget, 'load_defaults', lambda: {})
    kb.init_db()
    with kbc.connect_closing() as conn:
        yield conn


def exhausted(db):
    root = kb.create_task(db, title='D3 root', body='budget_tokens: 100\nbudget_seconds: 1000')
    run = kb.claim_task(db, root)
    budget.record_usage(db, root, run.current_run_id, 150)
    db.execute("UPDATE tasks SET status='ready' WHERE id=?", (root,))
    db.execute('UPDATE kanban_tree_budgets SET exceeded=1 WHERE root_id=?', (root,))
    db.execute('UPDATE task_runs SET ended_at=started_at WHERE task_id=?', (root,))
    db.commit()
    return root


def test_exceeded_claim_parks_once_and_alerts_with_cause(db, tmp_path):
    root = exhausted(db)
    causes = task_causes(db, 'probe')
    assert root in causes[0] and 'usado 150 de 100' in causes[0]
    assert kb.claim_task(db, root) is None
    assert kb.get_task(db, root).status == 'blocked'
    assert kb.get_task(db, root).block_kind == 'budget_exceeded'
    assert kb.claim_task(db, root) is None
    budget.enforce(db)
    budget.enforce(db)
    comments = [c.body for c in kb.list_comments(db, root) if c.body.startswith('budget_exceeded:')]
    assert len(comments) == 1
    assert 'usado 150 de 100' in comments[0]
    sent = []
    send = lambda text: sent.append(text) or True
    path = tmp_path / 'health.json'
    for now in (100, 999, 1000):
        observe(path, now=now, ready=True, spawned=False, causes=causes, send=send)
    assert not sent
    observe(path, now=1001, ready=True, spawned=False, causes=causes, send=send)
    for now in (2000, 3000):
        observe(path, now=now, ready=True, spawned=False, causes=causes, send=send)
    assert len(sent) == 1 and root in sent[0] and 'budget_exceeded' in sent[0]


def test_enforcement_comments_each_ready_descendant_once(db):
    root = exhausted(db)
    child = kb.create_task(db, title='D3 child', creator_task_id=root)
    budget.enforce(db)
    budget.enforce(db)
    for tid in (root, child):
        assert kb.get_task(db, tid).status == 'blocked'
        comments = [c for c in kb.list_comments(db, tid) if c.body.startswith('budget_exceeded:')]
        assert len(comments) == 1


@pytest.mark.parametrize('body', ['budget_tokens: 120', 'budget_seconds: 1'])
def test_lower_than_spent_rejected_atomically(db, body):
    root = exhausted(db)
    db.execute('UPDATE task_runs SET ended_at=started_at+2 WHERE task_id=?', (root,))
    db.commit()
    previous = kb.get_task(db, root).body
    with pytest.raises(ValueError, match='gasto actual'):
        kb.edit_task(db, root, body=body)
    assert kb.get_task(db, root).body == previous
    assert budget.limits(db, root)['budget_tokens'] == 100


def test_raise_cap_keeps_history_and_clears_flag(db):
    root = exhausted(db)
    assert kb.edit_task(db, root, body='budget_tokens: 1050\nbudget_seconds: 1000')
    assert budget.consumption(db, root)['tokens'] == 150
    assert budget.limits(db, root)['budget_tokens'] == 1050
    assert budget.limits(db, root)['exceeded'] == 0


def test_alert_failure_not_resent_and_recovery_rearms(tmp_path):
    path = tmp_path / 'health.json'
    sent = []
    def send(text):
        sent.append(text)
        return False
    for now in (0, 901, 1000):
        state = observe(path, now=now, ready=True, spawned=False, causes=['probe: credenciales/cuota'], send=send)
    assert len(sent) == 1 and not state['alert_sent'] and state['alert_attempted']
    observe(path, now=1100, ready=True, spawned=True, causes=[], send=send)
    for now in (1200, 2101):
        observe(path, now=now, ready=True, spawned=False, causes=['probe: perfil inexistente'], send=send)
    assert len(sent) == 2


def test_cli_rejects_cap_below_historical_consumption(db, capsys):
    from argparse import Namespace
    from hermes_cli.kanban import kanban_command
    root = exhausted(db)
    rc = kanban_command(Namespace(kanban_action='edit', task_id=root,
                                  title=None, body='budget_tokens: 120', priority=None,
                                  result=None, summary=None, metadata=None))
    assert rc != 0
    assert 'gasto actual 150' in capsys.readouterr().err
    assert budget.limits(db, root)['budget_tokens'] == 100


def test_causes_classify_missing_profile_auth_and_review(db, monkeypatch):
    monkeypatch.setattr('hermes_cli.profiles.profile_exists', lambda name: name == 'default')
    root = kb.create_task(db, title='profile removed after assignment')
    db.execute("UPDATE tasks SET assignee='removed-profile' WHERE id=?", (root,))
    auth = kb.create_task(db, title='auth')
    db.execute("UPDATE tasks SET assignee='default',last_failure_error='unauthorized' WHERE id=?", (auth,))
    rev = kb.create_task(db, title='review')
    db.execute("UPDATE tasks SET assignee='default',status='review' WHERE id=?", (rev,))
    db.commit()
    causes = task_causes(db, 'probe')
    assert any(root in c and 'perfil inexistente' in c for c in causes)
    assert any(auth in c and 'credenciales/cuota' in c for c in causes)
    assert any(rev in c for c in causes)

"""Aggregate fleet budgets and independent evidence review contracts."""
import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_tree_budget as budget
from hermes_cli import kanban_cross_review as review


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    kb.init_db()
    with kbc.connect_closing() as db:
        yield db


def test_budget_sums_retries_children_and_warns_once(conn, monkeypatch):
    monkeypatch.setattr(budget, 'load_defaults', lambda: {})
    root = kb.create_task(conn, title='root', body='budget_tokens: 100\nbudget_seconds: 1000')
    child = kb.create_task(conn, title='child', creator_task_id=root)
    first = kb.claim_task(conn, child)
    budget.record_usage(conn, child, first.current_run_id, 81)
    budget.enforce(conn)
    budget.enforce(conn)
    assert len([c for c in kb.list_comments(conn, root) if '80%' in c.body]) == 1
    kb.complete_task(conn, child, result='first', expected_run_id=first.current_run_id)
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child,))
    conn.commit()
    second = kb.claim_task(conn, child)
    budget.record_usage(conn, child, second.current_run_id, 20)
    assert root in budget.enforce(conn)
    assert kb.get_task(conn, root).block_kind == 'budget_exceeded'
    assert kb.get_task(conn, child).status == 'blocked'
    assert kb.claim_task(conn, child) is None
    assert not kb.complete_task(conn, root, result='cannot bypass', force=True)
    assert conn.execute("SELECT COUNT(*) FROM fleet_supervisor_inbox WHERE key LIKE 'budget_exceeded-%'").fetchone()[0] == 1


def test_seconds_sum_parallel_runs_and_type_defaults(conn, monkeypatch):
    monkeypatch.setattr(budget, 'load_defaults', lambda: {'research': {'budget_tokens': 500, 'budget_seconds': 100}})
    root = kb.create_task(conn, title='root', body='task_type: research')
    a = kb.create_task(conn, title='a', creator_task_id=root)
    b = kb.create_task(conn, title='b', creator_task_id=root)
    kb.claim_task(conn, a)
    kb.claim_task(conn, b)
    conn.execute('UPDATE task_runs SET started_at=100 WHERE task_id IN (?, ?)', (a, b))
    conn.commit()
    assert budget.consumption(conn, root, now=151)['seconds'] == 102
    assert root in budget.enforce(conn, now=151)


def test_decomposition_is_lineage_dependency_is_not(conn):
    root = kb.create_task(conn, title='decompose', triage=True)
    from hermes_cli.kanban_db_graph import decompose_triage_task
    children = decompose_triage_task(conn, root, root_assignee=None, children=[{'title': 'part'}])
    assert budget.root_for(conn, children[0]) == root
    unrelated = kb.create_task(conn, title='dependency', parents=children)
    assert budget.root_for(conn, unrelated) == unrelated


@pytest.mark.parametrize('kind', ['send_email', 'publish_post', 'listing', 'skill_change'])
def test_cross_review_is_independent_and_pass_closes(conn, monkeypatch, kind):
    monkeypatch.setattr(review, 'fleet_profiles', lambda: ['fleetfabric', 'fleetliterature', 'fleetops'])
    monkeypatch.setattr('hermes_cli.kanban_evidence.verify_evidence', lambda task, result: None)
    root = kb.create_task(conn, title=kind, assignee='fleetfabric', body='task_type: '+kind)
    assert not kb.complete_task(conn, root, result=json.dumps({'url': 'https://example.com'}))
    assert kb.get_task(conn, root).status == 'review'
    row = conn.execute('SELECT * FROM kanban_cross_reviews WHERE original_id=?', (root,)).fetchone()
    child = kb.get_task(conn, row['review_id'])
    assert child.assignee != 'fleetfabric'
    assert budget.root_for(conn, child.id) == root
    assert 'solo lectura' in child.body
    assert not kb.complete_task(conn, root, force=True)
    run = kb.claim_task(conn, child.id)
    assert kb.complete_task(conn, child.id, result=json.dumps({'verdict': 'PASS', 'proof': 'Primary evidence checked via read-only API'}), expected_run_id=run.current_run_id)
    assert kb.get_task(conn, root).status == 'done'


@pytest.mark.parametrize('verdict,status', [('FAIL','ready'), ('NO_VERIFICADO','review')])
def test_review_verdict_routes_original(conn, monkeypatch, verdict, status):
    monkeypatch.setattr(review, 'fleet_profiles', lambda: ['fleetfabric', 'fleetliterature'])
    monkeypatch.setattr('hermes_cli.kanban_evidence.verify_evidence', lambda task, result: None)
    original = kb.create_task(conn, title='post', assignee='fleetliterature', body='task_type: publish_post')
    assert not kb.complete_task(conn, original, result='url: https://example.com')
    rid = conn.execute('SELECT review_id FROM kanban_cross_reviews WHERE original_id=?', (original,)).fetchone()[0]
    assert not kb.complete_task(conn, rid, result='verdict: PASS')
    assert kb.complete_task(conn, rid, result=json.dumps({'verdict': verdict, 'proof': 'Observed HTTP failure'}))
    assert kb.get_task(conn, original).status == status
    assert kb.get_task(conn, original).assignee == 'fleetliterature'


def test_provider_accounting_includes_delegate_and_cache_tokens(conn, monkeypatch):
    from types import SimpleNamespace
    from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
    root = kb.create_task(conn, title='provider budget', body='budget_tokens: 100')
    run = kb.claim_task(conn, root)
    path = next(r[2] for r in conn.execute('PRAGMA database_list') if r[1] == 'main')
    monkeypatch.setenv('HERMES_KANBAN_TASK', root)
    monkeypatch.setenv('HERMES_KANBAN_RUN_ID', str(run.current_run_id))
    monkeypatch.setenv('HERMES_KANBAN_DB', path)
    interrupts = []
    agent = SimpleNamespace(interrupt=interrupts.append)
    budget.record_response(agent, 30)
    with delegated_child_context('child-session'):
        budget.record_response(agent, 71)
        with pytest.raises(PermissionError):
            kb.create_task(conn, title='forbidden model mutation')
    assert budget.consumption(conn, root)['tokens'] == 101
    assert interrupts == ['budget_exceeded']
    with non_dispatcher_owned_context():
        budget.record_response(agent, 900)
    assert budget.consumption(conn, root)['tokens'] == 101


def test_config_loader_and_task_override(conn, tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('kanban:\n  budget_defaults:\n    budget_tokens: 900\n    budget_seconds: 700\n    research:\n      budget_tokens: 300\n')
    root = kb.create_task(conn, title='configured', body='task_type: research\nbudget_seconds: 50')
    with budget.accounting_txn(conn):
        cap = budget.limits(conn, root)
    assert (cap['budget_tokens'], cap['budget_seconds']) == (300, 50)
    path = next(r[2] for r in conn.execute('PRAGMA database_list') if r[1] == 'main')
    from pathlib import Path
    assert Path(path+budget.BACKUP_SUFFIX).exists()


def test_review_request_cannot_reassign_original_or_self_review(conn, monkeypatch):
    monkeypatch.setattr(review, 'fleet_profiles', lambda: ['fleetfabric', 'fleetliterature'])
    monkeypatch.setattr('hermes_cli.kanban_evidence.verify_evidence', lambda task, result: None)
    original = kb.create_task(conn, title='skill', assignee='fleetfabric', body='task_type: skill_change')
    assert kb.request_review(conn, original, summary='diff: changed\nregression: passed', reviewer='fleetfabric')
    assert kb.get_task(conn, original).assignee == 'fleetfabric'
    assert kb.claim_review_task(conn, original) is None
    assert not kb.complete_task(conn, original, force=True)
    assert conn.execute('SELECT reviewer FROM kanban_cross_reviews WHERE original_id=?', (original,)).fetchone()[0] == 'fleetliterature'


def test_pass_reconciles_after_completion_interruption(conn, monkeypatch):
    monkeypatch.setattr(review, 'fleet_profiles', lambda: ['fleetfabric', 'fleetliterature'])
    monkeypatch.setattr('hermes_cli.kanban_evidence.verify_evidence', lambda task, result: None)
    original = kb.create_task(conn, title='post', assignee='fleetfabric', body='task_type: publish_post')
    kb.complete_task(conn, original, result='url: https://example.com')
    rid = conn.execute('SELECT review_id FROM kanban_cross_reviews WHERE original_id=?', (original,)).fetchone()[0]
    complete = kb.complete_task
    def interrupted(db, tid, **kwargs):
        if tid == original:
            raise RuntimeError('simulated crash after verdict commit')
        return complete(db, tid, **kwargs)
    monkeypatch.setattr(kb, 'complete_task', interrupted)
    with pytest.raises(RuntimeError, match='simulated crash'):
        complete(conn, rid, result='verdict: PASS\nproof: primary response checked')
    assert kb.get_task(conn, original).status == 'review'
    monkeypatch.setattr(kb, 'complete_task', complete)
    review.reconcile(conn)
    assert kb.get_task(conn, original).status == 'done'


def test_type_and_reviewed_evidence_cannot_be_removed(conn, monkeypatch):
    monkeypatch.setattr(review, 'fleet_profiles', lambda: ['fleetfabric', 'fleetliterature'])
    monkeypatch.setattr('hermes_cli.kanban_evidence.verify_evidence', lambda task, result: None)
    original = kb.create_task(conn, title='post', body='task_type: publish_post')
    with pytest.raises(ValueError, match='task_type'):
        kb.edit_task(conn, original, body='task_type: generic')
    kb.complete_task(conn, original, result='url: https://example.com')
    with pytest.raises(ValueError, match='immutable'):
        kb.edit_task(conn, original, result='url: https://example.org')


def test_budget_terminates_real_owned_workers(conn):
    import subprocess
    import sys
    from hermes_cli import kanban_db_dispatch as dispatch
    root = kb.create_task(conn, title='owned worker', body='budget_tokens: 1')
    run = kb.claim_task(conn, root)
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        dispatch._set_worker_pid(conn, root, process.pid)
        budget.record_usage(conn, root, run.current_run_id, 2)
        assert budget.enforce(conn) == [root]
        assert process.wait(timeout=5) != 0
        assert kb.get_task(conn, root).block_kind == 'budget_exceeded'
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_swarm_workers_verifier_and_synthesizer_share_root(conn):
    from hermes_cli.kanban_swarm import create_swarm, SwarmWorkerSpec
    swarm = create_swarm(conn, goal='review and synthesize',
                         workers=[SwarmWorkerSpec(profile='fleetfabric', title='worker', body='work')],
                         verifier_assignee='fleetliterature', synthesizer_assignee='fleetops')
    for tid in [*swarm.worker_ids, swarm.verifier_id, swarm.synthesizer_id]:
        assert budget.root_for(conn, tid) == swarm.root_id
    run = kb.claim_task(conn, swarm.worker_ids[0])
    budget.record_usage(conn, run.id, run.current_run_id, 123)
    assert budget.consumption(conn, swarm.root_id)['tokens'] == 123


def test_real_response_hook_records_canonical_cache_usage(conn, monkeypatch):
    from types import SimpleNamespace
    from run_agent import AIAgent
    from agent.turn_usage import record_response_usage
    root = kb.create_task(conn, title='actual response hook')
    run = kb.claim_task(conn, root)
    path = next(r[2] for r in conn.execute('PRAGMA database_list') if r[1] == 'main')
    monkeypatch.setenv('HERMES_KANBAN_TASK', root)
    monkeypatch.setenv('HERMES_KANBAN_RUN_ID', str(run.current_run_id))
    monkeypatch.setenv('HERMES_KANBAN_DB', path)
    agent = AIAgent(api_key='test-only', base_url='https://example.com/v1', provider='nous',
                    api_mode='chat_completions', model='test-model', session_id='budget-test', platform='cli',
                    quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
                    enabled_toolsets=['file'])
    try:
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=7, total_tokens=107,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=40, cache_write_tokens=10),
                                completion_tokens_details=None)
        record_response_usage(agent, SimpleNamespace(usage=usage, model='test-model'),
                              messages=[{'role': 'user', 'content': 'test'}], api_call_count=1,
                              api_duration=.1, compression_attempts=0, max_compression_attempts=3)
        assert budget.consumption(conn, root)['tokens'] == 107
    finally:
        agent.close()


def test_reopen_resubmit_restores_review_and_contract_changes_need_new_reviewer(conn, monkeypatch):
    monkeypatch.setattr(review, 'fleet_profiles', lambda: ['fleetfabric', 'fleetliterature', 'fleetops'])
    monkeypatch.setattr('hermes_cli.kanban_evidence.verify_evidence', lambda task, result: None)
    original = kb.create_task(conn, title='post', assignee='fleetfabric', body='task_type: publish_post')
    kb.complete_task(conn, original, result='url: https://example.com')
    first = conn.execute('SELECT review_id FROM kanban_cross_reviews WHERE original_id=?', (original,)).fetchone()[0]
    assert kb.reopen_review_task(conn, original)
    assert not kb.complete_task(conn, original)
    assert kb.get_task(conn, original).status == 'review'
    assert kb.reopen_review_task(conn, original)
    kb.edit_task(conn, original, body='task_type: publish_post\nAdditional acceptance criterion')
    assert not kb.complete_task(conn, original)
    second = conn.execute('SELECT review_id FROM kanban_cross_reviews WHERE original_id=? ORDER BY rowid DESC', (original,)).fetchone()[0]
    assert first != second
    assert kb.complete_task(conn, first, result='verdict: PASS\nproof: old criteria observed')
    assert kb.get_task(conn, original).status == 'review'
    assert kb.complete_task(conn, second, result='verdict: PASS\nproof: new criteria observed')
    assert kb.get_task(conn, original).status == 'done'


def test_budget_notification_retries_after_crash(conn, monkeypatch):
    root = kb.create_task(conn, title='crash recovery', body='budget_tokens: 1')
    run = kb.claim_task(conn, root)
    budget.record_usage(conn, root, run.current_run_id, 2)
    notify = budget.notify_alfred
    def interrupted(*args, **kwargs):
        raise RuntimeError('interrupted delivery')
    monkeypatch.setattr(budget, 'notify_alfred', interrupted)
    with pytest.raises(RuntimeError, match='interrupted delivery'):
        budget.enforce(conn)
    assert kb.get_task(conn, root).block_kind == 'budget_exceeded'
    monkeypatch.setattr(budget, 'notify_alfred', notify)
    budget.enforce(conn)
    budget.enforce(conn)
    assert conn.execute("SELECT COUNT(*) FROM tasks WHERE idempotency_key=?", ('budget_exceeded-'+root,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM fleet_supervisor_inbox WHERE key=?", ('budget_exceeded-'+root,)).fetchone()[0] == 1


@pytest.mark.parametrize('terminal', ['done', 'archived'])
def test_budget_enforcement_does_not_resurrect_terminal_root(conn, terminal):
    root = kb.create_task(conn, title='completed diagnosis', body='budget_tokens: 1')
    run = kb.claim_task(conn, root)
    budget.record_usage(conn, root, run.current_run_id, 2)
    with kb.write_txn(conn):
        kb._end_run(conn, root, outcome='completed', status='done')
        conn.execute('UPDATE tasks SET status=? WHERE id=?', (terminal, root))
    budget.enforce(conn)
    assert kb.get_task(conn, root).status == terminal

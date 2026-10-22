"""Fleet acceptance is verified at completion, including review and legacy cards."""
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_evidence as evidence


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    kb.create_board(slug='alfred-fleet', name='Fleet')
    with kbc.connect_closing(board='alfred-fleet') as db:
        yield db


def test_missing_legacy_evidence_goes_to_review(conn):
    tid = kb.create_task(conn, title='legacy')
    assert not kb.complete_task(conn, tid, result='I did it')
    assert kb.get_task(conn, tid).status == 'review'
    assert 'generic requires' in kb.list_comments(conn, tid)[-1].body
    assert not kb.complete_task(conn, tid, force=True)


@pytest.mark.parametrize('kind,result', [
    ('send_email', {'gmail_message_id': '19a987a654321bcf', 'sender': 'sender@example.com'}),
    ('publish_post', {'publication_id': '12345'}),
    ('listing', {'url': 'https://example.com/profile'}),
    ('code_change', {'commit_sha': 'a'*40, 'remote_url': 'https://example.com/repo.git'}),
    ('research', {'artifact_path': '__artifact__'}),
    ('generic', {'artifact_path': '__artifact__'}),
])
def test_type_requires_correct_evidence(conn, tmp_path, monkeypatch, kind, result):
    artifact = tmp_path / 'deliverable.md'
    artifact.write_text('observed effect')
    result = {k: str(artifact) if v == '__artifact__' else v for k,v in result.items()}
    monkeypatch.setattr(evidence, 'reachable_url', lambda value: value == 'https://example.com/profile')
    monkeypatch.setattr(evidence, 'remote_commit', lambda value: value.get('commit_sha') == 'a'*40)
    tid = kb.create_task(conn, title=kind, body='task_type: '+kind)
    assert not kb.complete_task(conn, tid, result='{}')
    needs_review = kind in {'send_email', 'publish_post', 'listing'}
    assert kb.complete_task(conn, tid, result=json.dumps(result)) is (not needs_review)
    assert kb.get_task(conn, tid).status == ('review' if needs_review else 'done')


def test_existing_artifact_is_checked_and_schema_invalid_rejected(conn, tmp_path):
    path = tmp_path / 'missing.md'
    tid = kb.create_task(conn, title='research', body='task_type: research')
    assert not kb.complete_task(conn, tid, result=json.dumps({'path': str(path)}))
    path.write_text('real')
    assert not kb.complete_task(conn, tid, result=json.dumps({'path': str(path), 'schema_valid': False}))
    assert kb.complete_task(conn, tid, result=str(path))


def test_owner_conflict_creation_assignment_and_claim(conn):
    first = kb.create_task(conn, title='first', body='resource_key: lead-1', assignee='alpha')
    second = kb.create_task(conn, title='second', body='resource_key: lead-1', assignee='beta')
    assert kb.get_task(conn, first).status == 'ready'
    assert kb.get_task(conn, second).block_kind == 'owner_conflict'
    assert kb.claim_task(conn, second) is None
    same = kb.create_task(conn, title='same', body='resource_key: lead-1', assignee='alpha')
    assert kb.get_task(conn, same).status == 'ready'
    kb.assign_task(conn, same, 'beta')
    assert kb.get_task(conn, same).status == 'blocked'
    conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (second,))
    assert kb.claim_task(conn, second) is None


def test_unknown_type_rejected(conn):
    with pytest.raises(ValueError, match='task_type'):
        kb.create_task(conn, title='typo', body='task_type: typo')


def test_completion_ownership_not_bypassed(conn, tmp_path, monkeypatch):
    tid = kb.create_task(conn, title='claimed', body='task_type: research')
    claimed = kb.claim_task(conn, tid)
    assert not kb.complete_task(conn, tid, result='missing', expected_run_id=claimed.current_run_id+1)
    assert kb.get_task(conn, tid).status == 'running'
    monkeypatch.setattr(kb, '_claim_is_live', lambda row: True)
    with pytest.raises(kb.LiveClaimError):
        kb.complete_task(conn, tid, result='missing')


def test_url_validation_rejects_private_and_credentials(monkeypatch):
    monkeypatch.setattr(evidence.socket, 'getaddrinfo', lambda *a: [(2, 1, 6, '', ('127.0.0.1', 80))])
    assert not evidence.reachable_url('http://localhost/private')
    assert not evidence.reachable_url('https://user:password@example.com')


def test_remote_sha_requires_remote_advertisement(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(evidence, '_public_url', lambda url: url)
    monkeypatch.setattr(evidence.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='a'*40+'\trefs/heads/main\n'))
    assert evidence.remote_commit({'commit_sha': 'a'*40, 'remote_url': 'https://example.com/repo'})
    assert not evidence.remote_commit({'commit_sha': 'b'*40, 'remote_url': 'https://example.com/repo'})
    assert not evidence.remote_commit({'commit_sha': 'a'*40, 'remote_url': '/local/repo'})


def test_sticky_owner_conflict_survives_recompute_and_review_claim(conn):
    first = kb.create_task(conn, title='first', body='resource_key: account', assignee='alpha')
    second = kb.create_task(conn, title='second', body='resource_key: account', assignee='beta')
    kb.recompute_ready(conn)
    assert kb.get_task(conn, second).status == 'blocked'
    conn.execute("UPDATE tasks SET status='review' WHERE id=?", (second,))
    assert kb.claim_review_task(conn, second) is None
    assert kb.get_task(conn, first).status == 'ready'


def test_disposable_scratch_artifact_cannot_close(conn, tmp_path):
    path = tmp_path/'scratch'/'result.txt'
    path.parent.mkdir()
    path.write_text('would be deleted')
    tid = kb.create_task(conn, title='scratch', body='task_type: research', workspace_path=str(path.parent))
    assert not kb.complete_task(conn, tid, result=str(path))
    assert 'disposable' in kb.list_comments(conn, tid)[-1].body


def test_done_evidence_cannot_be_overwritten_or_retyped(conn, tmp_path):
    path = tmp_path/'persist.txt'
    path.write_text('keep this')
    tid = kb.create_task(conn, title='done', body='task_type: research')
    assert kb.complete_task(conn, tid, result=str(path))
    with pytest.raises(ValueError, match='cannot be invalidated'):
        kb.edit_task(conn, tid, result='finished')
    with pytest.raises(ValueError, match='cannot be invalidated'):
        kb.edit_task(conn, tid, body='task_type: send_email')
    assert kb.get_task(conn, tid).status == 'done'
    assert kb.get_task(conn, tid).result == str(path)


def test_evidence_checked_against_concurrent_body_edit(conn, tmp_path, monkeypatch):
    path = tmp_path/'persist.txt'
    path.write_text('keep this')
    tid = kb.create_task(conn, title='race', body='task_type: research')
    verify = evidence.verify_evidence
    def concurrent_change(task, result):
        outcome = verify(task, result)
        conn.execute("UPDATE tasks SET body='task_type: send_email' WHERE id=?", (tid,))
        return outcome
    monkeypatch.setattr(evidence, 'verify_evidence', concurrent_change)
    assert not kb.complete_task(conn, tid, result=str(path))
    assert kb.get_task(conn, tid).status != 'done'


def test_review_uses_stored_verifiable_result(conn, tmp_path):
    path = tmp_path/'persist.txt'
    path.write_text('kept')
    tid = kb.create_task(conn, title='review', body='task_type: research')
    conn.execute("UPDATE tasks SET status='review', result=? WHERE id=?", (str(path),tid))
    assert kb.complete_task(conn, tid)
    assert kb.get_task(conn, tid).result == str(path)

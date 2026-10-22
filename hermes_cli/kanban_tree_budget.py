"""Durable tree accounting; iteration limits reset per call, not per task tree.

Seconds are summed worker-run wall time, including parallel runs and retries.
Tokens use canonical provider totals (input + cache + output), never estimates.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from contextlib import contextmanager

BACKUP_SUFFIX = '.bak-pre-i1-20261022'


def ensure_schema(conn):
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name='kanban_tree_budgets'").fetchone():
        return
    # Separate read connection works even when creation owns a write transaction.
    for row in conn.execute('PRAGMA database_list'):
        if row[1] == 'main' and row[2]:
            path = Path(row[2])
            backup = Path(str(path) + BACKUP_SUFFIX)
            if not backup.exists():
                with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as source:
                    with sqlite3.connect(backup) as target:
                        source.backup(target)
    conn.execute('CREATE TABLE IF NOT EXISTS kanban_tree_budgets ('
                 'root_id TEXT PRIMARY KEY, budget_tokens INTEGER NOT NULL, '
                 'budget_seconds INTEGER NOT NULL, warned INTEGER NOT NULL DEFAULT 0, '
                 'exceeded INTEGER NOT NULL DEFAULT 0)')
    conn.execute('CREATE TABLE IF NOT EXISTS kanban_run_usage ('
                 'run_id INTEGER PRIMARY KEY, task_id TEXT NOT NULL, tokens INTEGER NOT NULL DEFAULT 0, '
                 'missing_usage INTEGER NOT NULL DEFAULT 0)')


def load_defaults():
    from hermes_cli.config_effective import load_user_config_effective
    return load_user_config_effective().get('kanban', {}).get('budget_defaults', {})


def root_for(conn, task_id):
    """Creation provenance survives unlinking. Dependencies are not ancestry."""
    seen = set()
    while task_id not in seen:
        seen.add(task_id)
        row = conn.execute("SELECT payload FROM task_events WHERE task_id=? AND kind='created' ORDER BY id LIMIT 1", (task_id,)).fetchone()
        data = json.loads(row[0] or '{}') if row else {}
        parent = data.get('creator_task_id') or data.get('from_decompose_of')
        if not parent or not conn.execute('SELECT 1 FROM tasks WHERE id=?', (parent,)).fetchone():
            return task_id
        task_id = parent
    raise ValueError('cyclic task creation lineage')


def limits(conn, root_id):
    from hermes_cli.kanban_evidence import fields
    ensure_schema(conn)
    row = conn.execute('SELECT * FROM kanban_tree_budgets WHERE root_id=?', (root_id,)).fetchone()
    if row:
        return dict(row)
    body = conn.execute('SELECT body FROM tasks WHERE id=?', (root_id,)).fetchone()
    values = fields(body[0] if body else '')
    defaults = load_defaults()
    typed = defaults.get(values.get('task_type', 'generic'), {})
    result = {}
    for key, fallback in [('budget_tokens', 400_000), ('budget_seconds', 1800)]:
        value = int(values.get(key, typed.get(key, defaults.get(key, fallback))))
        if value <= 0:
            raise ValueError(key + ' must be positive')
        result[key] = value
    conn.execute('INSERT OR IGNORE INTO kanban_tree_budgets(root_id,budget_tokens,budget_seconds) VALUES (?,?,?)',
                 (root_id, result['budget_tokens'], result['budget_seconds']))
    return dict(conn.execute('SELECT * FROM kanban_tree_budgets WHERE root_id=?', (root_id,)).fetchone())



def _run_elapsed(row, now):
    end = row['ended_at']
    # A watchdog can park a card while its worker remains alive. Its time
    # budget must keep running until that process exits or the budget stops it.
    if end is not None and str(row['summary'] or '').startswith('stalled:'):
        from hermes_cli import kanban_db as kb
        if (str(row['claim_lock'] or '').startswith(kb._host_prefix())
                and kb._worker_alive(row['worker_pid'], row['worker_started_at'])):
            end = None
    return max(0, (now if end is None else end) - row['started_at'])


def consumption(conn, root_id, *, now=None):
    ensure_schema(conn)
    now = int(time.time()) if now is None else now
    ids = {row[0] for row in conn.execute('SELECT id FROM tasks') if root_for(conn, row[0]) == root_id}
    runs = [row for row in conn.execute('SELECT * FROM task_runs') if row['task_id'] in ids]
    usage = [row for row in conn.execute('SELECT * FROM kanban_run_usage') if row['task_id'] in ids]
    tokens = sum(row['tokens'] for row in usage)
    seconds = sum(_run_elapsed(row, now) for row in runs)
    metered = {row['run_id'] for row in usage}
    return {'tokens': tokens, 'seconds': seconds, 'task_ids': sorted(ids),
            'missing_usage_calls': sum(row['missing_usage'] for row in usage),
            'unmetered_runs': sum(row['id'] not in metered for row in runs)}


@contextmanager
def accounting_txn(conn):
    """Internal provider ledger, independent of model-accessible mutation rights."""
    nested = conn.in_transaction
    if not nested:
        conn.execute('BEGIN IMMEDIATE')
    try:
        yield
        if not nested:
            conn.commit()
    except BaseException:
        if not nested:
            conn.rollback()
        raise


def record_usage(conn, task_id, run_id, tokens, *, missing=False):
    with accounting_txn(conn):
        ensure_schema(conn)
        if not conn.execute('SELECT 1 FROM task_runs WHERE id=? AND task_id=?', (run_id, task_id)).fetchone():
            raise ValueError('usage requires a real task run')
        conn.execute('INSERT INTO kanban_run_usage(run_id,task_id,tokens,missing_usage) VALUES (?,?,?,?) '
                     'ON CONFLICT(run_id) DO UPDATE SET tokens=tokens+excluded.tokens, '
                     'missing_usage=missing_usage+excluded.missing_usage',
                     (run_id, task_id, max(0, int(tokens)), int(missing)))
        root = root_for(conn, task_id)
        warn(conn, root, limits(conn, root), consumption(conn, root), int(time.time()))


def notify_alfred(conn, root_id, reason, details):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_progress import _ensure_schema, _flush_inbox
    _ensure_schema(conn)
    key = reason + '-' + root_id
    body = json.dumps({'recipient': 'alfred', 'task_id': root_id, 'reason': reason, **details})
    with kb.write_txn(conn):
        conn.execute('INSERT OR REPLACE INTO fleet_supervisor_inbox VALUES (?,?)', (key, body))
    _flush_inbox(conn)
    if conn.execute('SELECT 1 FROM tasks WHERE idempotency_key=?', (key,)).fetchone():
        return
    if conn.execute("SELECT 1 FROM tasks WHERE id=? AND created_by='alfred-tree-policy'", (root_id,)).fetchone():
        return
    kb.create_task(conn, title='Alfred: '+reason+' '+root_id, assignee='default',
                   created_by='alfred-tree-policy', idempotency_key=key,
                   body='task_type: research\nAviso interno para Alfred: '+body+'\n'
                        'Diagnosticar y registrar un artefacto durable. No repetir efectos externos ni enviar mensajes.')


def warn(conn, root, cap, used, now):
    if max(used['tokens'] / cap['budget_tokens'], used['seconds'] / cap['budget_seconds']) >= .8 and not cap['warned']:
        conn.execute('INSERT INTO task_comments(task_id,author,body,created_at) VALUES (?,?,?,?)',
                     (root, 'alfred', 'Budget 80%: '+json.dumps(used), now))
        conn.execute('UPDATE kanban_tree_budgets SET warned=1 WHERE root_id=?', (root,))


def enforce(conn, *, now=None, signal_fn=None):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_dispatch import _terminate_reclaimed_worker, UNVERIFIED_WORKER_FINGERPRINT
    now = int(time.time()) if now is None else now
    blocked = []
    with kb.write_txn(conn):
        ensure_schema(conn)
    tracked = conn.execute("SELECT id FROM tasks WHERE status NOT IN ('done','archived') UNION SELECT task_id FROM kanban_run_usage").fetchall()
    roots = {root_for(conn, row[0]) for row in tracked}
    for root in sorted(roots):
        with kb.write_txn(conn):
            cap = limits(conn, root)
            used = consumption(conn, root, now=now)
            ratio = max(used['tokens'] / cap['budget_tokens'], used['seconds'] / cap['budget_seconds'])
            warn(conn, root, cap, used, now)
            exceeded = bool(cap['exceeded'] or ratio > 1)
            if not exceeded:
                continue
            conn.execute('UPDATE kanban_tree_budgets SET exceeded=1 WHERE root_id=?', (root,))
            for task_id in used['task_ids']:
                row = conn.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
                if task_id == root or row['status'] not in {'done', 'archived'}:
                    conn.execute("UPDATE tasks SET status='blocked', block_kind='budget_exceeded', claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?", (task_id,))
                    kb._end_run(conn, task_id, outcome='budget_exceeded', status='blocked', summary='budget_exceeded')
            if not cap['exceeded']:
                kb._insert_comment(conn, root, 'alfred', 'budget_exceeded: '+json.dumps(used), now)
                kb._append_event(conn, root, 'budget_exceeded', used)
                blocked.append(root)
        # Signal only fingerprint-verified, host-local workers, retaining run identity
        # so a failed termination is retried next tick. Never signal by bare PID.
        for run in conn.execute('SELECT * FROM task_runs WHERE worker_pid IS NOT NULL').fetchall():
            if run['task_id'] in used['task_ids'] and run['worker_started_at'] not in (None, UNVERIFIED_WORKER_FINGERPRINT):
                _terminate_reclaimed_worker(run['worker_pid'], run['claim_lock'], started_at=run['worker_started_at'], signal_fn=signal_fn)
        notify_alfred(conn, root, 'budget_exceeded', used)
    return blocked


def allowed(conn, task_id):
    """Sticky root guard for manual claims and completion between ticks."""
    from hermes_cli import kanban_db as kb
    if not kb.get_task(conn, task_id):
        return True
    with accounting_txn(conn):
        cap = limits(conn, root_for(conn, task_id))
        used = consumption(conn, root_for(conn, task_id))
        return not cap['exceeded'] and used['tokens'] <= cap['budget_tokens'] and used['seconds'] <= cap['budget_seconds']


def record_response(agent, tokens, *, missing=False):
    """Includes in-process delegates; cron contexts deliberately carry no spend."""
    from agent.delegation_context import _NON_DISPATCHER_OWNED_CONTEXT
    task_id, run_id, path = (os.getenv(key) for key in ('HERMES_KANBAN_TASK', 'HERMES_KANBAN_RUN_ID', 'HERMES_KANBAN_DB'))
    if not (task_id and run_id and path) or _NON_DISPATCHER_OWNED_CONTEXT.get():
        return
    # Accounting is intentionally separate from model-accessible mutation tools.
    with sqlite3.connect(path, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        record_usage(conn, task_id, int(run_id), tokens, missing=missing)
        if not allowed(conn, task_id):
            agent.interrupt('budget_exceeded')

"""Durable, once-per-episode dispatcher alert. No model or raw error payloads."""
import json
import subprocess
import sys
from pathlib import Path


def send_alert(text):
    script = Path.home() / 'alfred/scripts/kanban_dispatch_alert.py'
    try:
        result = subprocess.run([sys.executable, str(script)], input=text,
                                text=True, capture_output=True, timeout=20, check=False)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def observe(path, *, now, ready, spawned, causes, send=send_alert):
    try:
        state = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        state = {}
    if not isinstance(state, dict) or not isinstance(state.get('since', 0), (int, float)):
        state = {}
    if not ready or spawned:
        state = {'active': False, 'since': None, 'alert_sent': False}
    else:
        if not state.get('active'):
            state = {'active': True, 'since': now, 'alert_sent': False}
        state['causes'] = causes
        if (now - state['since'] > 900 and not state.get('alert_attempted')):
            # Reserve before sending: process death cannot duplicate an alert.
            state['alert_attempted'] = True
            state['last_seen'] = now
            _save(path, state)
            text = 'Alfred: despachador sin lanzar workers por más de 15 minutos.\n'
            text += '\n'.join(causes) or 'Causa no confirmada; revisar el dispatcher.'
            state['alert_sent'] = bool(send(text[:3900]))
    state['last_seen'] = now
    _save(path, state)
    return state


def _save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def task_causes(conn, board, result=None):
    from hermes_cli import kanban_tree_budget as budget
    from hermes_cli import kanban_db_dispatch as dispatch
    from hermes_cli.profiles import profile_exists
    reasons = {}
    if result:
        for task_id, reason in result.respawn_guarded:
            reasons[task_id] = reason
        for task_id, _profile, _count in result.skipped_per_profile_capped:
            reasons[task_id] = 'profile_capacity'
    causes = []
    has_budget = conn.execute("SELECT 1 FROM sqlite_master WHERE name='kanban_tree_budgets'").fetchone()
    from hermes_cli.kanban_cross_review import managed_original
    statuses = "('ready','review')" if dispatch.review_dispatch_enabled() else "('ready')"
    for row in conn.execute(f"SELECT id, assignee, status FROM tasks WHERE status IN {statuses}"):
        task_id, profile = row['id'], row['assignee']
        if row['status'] == 'review' and managed_original(conn, task_id):
            continue
        root = budget.root_for(conn, task_id)
        cap_row = conn.execute('SELECT * FROM kanban_tree_budgets WHERE root_id=?', (root,)).fetchone() if has_budget else None
        cap = dict(cap_row) if cap_row else {'exceeded': 0, 'budget_tokens': float('inf'), 'budget_seconds': float('inf')}
        used = budget.consumption(conn, root) if has_budget else {'tokens': 0, 'seconds': 0}
        if cap['exceeded'] or used['tokens'] > cap['budget_tokens'] or used['seconds'] > cap['budget_seconds']:
            reason = (f"budget_exceeded: usado {used['tokens']} de {cap['budget_tokens']} tokens; "
                      f"{used['seconds']} de {cap['budget_seconds']} segundos")
        elif not profile:
            reason = 'sin assignee'
        elif not profile_exists(profile):
            reason = 'perfil inexistente (o lane externa sin worker Hermes)'
        else:
            failure = conn.execute('SELECT last_failure_error FROM tasks WHERE id=?', (task_id,)).fetchone()[0]
            reason = ('credenciales/cuota' if dispatch._RESPAWN_BLOCKER_RE.search(failure or '')
                      else reasons.get(task_id, 'causa no confirmada (spawn/capacidad/PATH)'))
        causes.append(f'{board}/{task_id}: {reason}')
    return causes

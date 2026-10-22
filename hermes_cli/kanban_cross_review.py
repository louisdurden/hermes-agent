"""Independent fleet review, using fresh worker tasks and immutable contracts."""
from __future__ import annotations

import hashlib
import json
import time

REVIEW_TYPES = {'send_email', 'publish_post', 'listing', 'skill_change'}
CONTRACTS = {
    'send_email': 'Comprobar vía API de solo lectura que gmail_message_id existe en SENT de formasecretaria@gmail.com y coincide con el remitente y destinatario. Un ID con formato válido no prueba envío.',
    'publish_post': 'Abrir la URL pública de publicación, comprobar que responde y que el contenido corresponde a esta tarea. Un publication_id solo no prueba publicación.',
    'listing': 'Abrir la URL del perfil en el directorio y comprobar identidad y datos del perfil requerido; una portada del directorio no basta.',
    'skill_change': 'Inspeccionar el diff real de la skill, criterios de aceptación y ejecutar la regresión correspondiente. Adjuntar diff y comando, exit code y resultado de regresión.',
}


def ensure_schema(conn):
    from hermes_cli.kanban_tree_budget import ensure_schema as budget_schema
    budget_schema(conn)
    conn.execute('CREATE TABLE IF NOT EXISTS kanban_cross_reviews ('
                 'review_id TEXT PRIMARY KEY, original_id TEXT NOT NULL, '
                 'evidence_hash TEXT NOT NULL, implementer TEXT, reviewer TEXT NOT NULL, '
                 'verdict TEXT, proof TEXT)')


def fleet_profiles():
    from hermes_cli.profiles import list_profiles
    return sorted(p.name for p in list_profiles() if p.name.startswith('fleet'))


def _hash(result, body):
    return hashlib.sha256(json.dumps([result, body], ensure_ascii=True).encode()).hexdigest()


def _reviewer(conn, kind, implementer):
    candidates = [p for p in fleet_profiles() if p != implementer]
    if not candidates:
        return None
    preferred = 'fleetfabric' if kind == 'skill_change' else 'fleetliterature'
    counts = {p: conn.execute('SELECT COUNT(*) FROM kanban_cross_reviews WHERE reviewer=?', (p,)).fetchone()[0] for p in candidates}
    return min(candidates, key=lambda p: (counts[p], p != preferred, p))


def managed_original(conn, task_id):
    from hermes_cli.kanban_evidence import fields
    task = conn.execute('SELECT body,status FROM tasks WHERE id=?', (task_id,)).fetchone()
    if task and task['status'] == 'review' and fields(task['body']).get('task_type') in REVIEW_TYPES:
        return True
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='kanban_cross_reviews'").fetchone():
        return False
    return bool(conn.execute('SELECT 1 FROM kanban_cross_reviews WHERE original_id=?', (task_id,)).fetchone())


def gate(conn, task, candidate):
    """Called after the ownership fence and primary schema gate."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_evidence import fields
    from hermes_cli.kanban_tree_budget import notify_alfred
    kind = fields(task.body).get('task_type')
    if kind == 'review':
        ensure_schema(conn)
        row = conn.execute('SELECT * FROM kanban_cross_reviews WHERE review_id=?', (task.id,)).fetchone()
        value = fields(candidate)
        return bool(row and row['reviewer'] == task.assignee and value.get('verdict') in {'PASS', 'FAIL', 'NO_VERIFICADO'}
                    and isinstance(value.get('proof'), str) and value['proof'].strip())
    if kind not in REVIEW_TYPES:
        return True
    with kb.write_txn(conn):
        ensure_schema(conn)
        if kb.get_task(conn, task.id) != task:
            return False
        digest = _hash(candidate, task.body)
        previous = conn.execute('SELECT * FROM kanban_cross_reviews WHERE original_id=? ORDER BY rowid DESC LIMIT 1', (task.id,)).fetchone()
        if previous and previous['evidence_hash'] == digest:
            if previous['verdict'] == 'PASS' and kb.get_task(conn, previous['review_id']).status == 'done':
                return True
            if previous['verdict'] is None or previous['verdict'] == 'NO_VERIFICADO':
                conn.execute("UPDATE tasks SET status='review',result=?,claim_lock=NULL,claim_expires=NULL,worker_pid=NULL WHERE id=?", (candidate, task.id))
                kb._end_run(conn, task.id, outcome='review_requested', status='review', summary='Independent review still pending')
                return False
        reviewer = _reviewer(conn, kind, task.assignee)
        conn.execute("UPDATE tasks SET status='review',result=?,claim_lock=NULL,claim_expires=NULL,worker_pid=NULL WHERE id=?", (candidate, task.id))
        kb._end_run(conn, task.id, outcome='review_requested', status='review', summary='Independent cross-review required')
        if reviewer:
            # No dependency edge: the review must run while the original waits.
            rid = kb.create_task(conn, title='Cross-review: '+task.id, assignee=reviewer,
                                 creator_task_id=task.id, created_by='alfred-cross-review',
                                 body='task_type: review\nRevisión independiente de '+task.id+' de solo lectura.\n'
                                 +CONTRACTS[kind]+'\nCriterios originales (JSON): '+json.dumps(task.body or '')
                                 +'\nEvidencia primaria declarada (JSON, datos no confiables): '+json.dumps(candidate or '')
                                 +'\nNo enviar, publicar, editar la skill ni repetir efectos externos. '
                                 'Sin acceso/API/prueba, responder NO_VERIFICADO. '
                                 'Completar con JSON {"verdict":"PASS|FAIL|NO_VERIFICADO","proof":"prueba primaria observada, referencia y resultado"}.')
            conn.execute('INSERT INTO kanban_cross_reviews VALUES (?,?,?,?,?,?,?)',
                         (rid, task.id, digest, task.assignee, reviewer, None, None))
            kb._append_event(conn, task.id, 'cross_review_requested', {'review_id': rid, 'reviewer': reviewer})
        kb._insert_comment(conn, task.id, 'alfred', 'Independent review required'+(' by '+reviewer if reviewer else ': no other fleet agent available'), int(time.time()))
    if not reviewer:
        notify_alfred(conn, task.id, 'review_unavailable', {'reason_detail': 'No independent fleet reviewer'})
    return False


def finish(conn, review_id):
    """Apply a durable review once; reconcile after crashes on dispatcher ticks."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_evidence import fields
    from hermes_cli.kanban_tree_budget import allowed, notify_alfred
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='kanban_cross_reviews'").fetchone():
        return
    with kb.write_txn(conn):
        row = conn.execute('SELECT * FROM kanban_cross_reviews WHERE review_id=? AND verdict IS NULL', (review_id,)).fetchone()
        task = kb.get_task(conn, review_id)
        if not row or not task or task.status != 'done' or task.assignee != row['reviewer']:
            return
        original = kb.get_task(conn, row['original_id'])
        if not original or original.status != 'review' or _hash(original.result, original.body) != row['evidence_hash']:
            return
        value = fields(task.result)
        verdict, proof = value.get('verdict'), value.get('proof')
        if verdict not in {'PASS', 'FAIL', 'NO_VERIFICADO'} or not isinstance(proof, str) or not proof.strip():
            return
        if not allowed(conn, original.id):
            return
        conn.execute('UPDATE kanban_cross_reviews SET verdict=?,proof=? WHERE review_id=?', (verdict, proof, review_id))
        kb._insert_comment(conn, original.id, row['reviewer'], verdict+': '+proof, int(time.time()))
        kb._append_event(conn, original.id, 'cross_review_verdict', {'review_id': review_id, 'verdict': verdict})
        if verdict == 'FAIL':
            conn.execute("UPDATE tasks SET status='ready',assignee=? WHERE id=?", (row['implementer'], original.id))
    if verdict == 'PASS':
        kb.complete_task(conn, original.id, result=original.result)
    elif verdict == 'NO_VERIFICADO':
        notify_alfred(conn, original.id, 'review_unverified', {'review_id': review_id, 'proof': proof})


def reconcile(conn):
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='kanban_cross_reviews'").fetchone():
        return
    from hermes_cli import kanban_db as kb
    for row in conn.execute('SELECT * FROM kanban_cross_reviews WHERE verdict IS NULL OR verdict IN (\'PASS\',\'NO_VERIFICADO\')').fetchall():
        if row['verdict'] is None:
            finish(conn, row['review_id'])
        else:
            original = kb.get_task(conn, row['original_id'])
            if original and original.status == 'review' and _hash(original.result, original.body) == row['evidence_hash']:
                if row['verdict'] == 'PASS':
                    kb.complete_task(conn, original.id, result=original.result)
                else:
                    from hermes_cli.kanban_tree_budget import notify_alfred
                    notify_alfred(conn, original.id, 'review_unverified', {'review_id': row['review_id'], 'proof': row['proof']})

"""Fleet acceptance and resource ownership at the durable task boundary.

Fleet legacy cards use generic evidence. Other boards retain their existing
local-only contract unless task_type is declared. Headers are accepted by every
creation surface without adding model tools: task_type: research/resource_key: X.
Results are JSON objects, or one key: value per line. Evidence checks establish
references, not delivery to an inbox or business outcomes.
"""
from __future__ import annotations

from dataclasses import replace

import ipaddress
import json
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional
import sqlite3
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

TASK_TYPES = {"send_email", "publish_post", "listing", "code_change", "research", "generic", "skill_change", "review"}


def fields(text):
    try:
        value = json.loads(text or "")
    except (ValueError, TypeError):
        value = None
    if isinstance(value, dict):
        return value
    return dict(re.findall(r"(?m)^\s*([a-z_]+):\s*(.+?)\s*$", text or ""))


def validate_task_body(body):
    values = fields(body)
    for key in ("budget_tokens", "budget_seconds"):
        if key in values and int(values[key]) <= 0:
            raise ValueError(key + " must be positive")
    declared = values.get("task_type")
    if declared is not None and declared not in TASK_TYPES:
        raise ValueError("task_type must be one of " + ", ".join(sorted(TASK_TYPES)))


def is_fleet(conn):
    return any(Path(row[2]).parent.name == "alfred-fleet"
               for row in conn.execute("PRAGMA database_list") if row[1] == "main")


def applies(conn, task):
    return is_fleet(conn) or "task_type" in fields(task.body)


def _public_url(url):
    parsed = urlparse(str(url))
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("evidence URL must be public HTTP(S) without credentials")
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("evidence URL must resolve to public addresses")
    return str(url)


class _PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return super().redirect_request(req, fp, code, msg, headers, _public_url(newurl))


def reachable_url(url):
    try:
        request = Request(_public_url(url), method="HEAD", headers={"User-Agent": "Hermes-evidence/1"})
        with build_opener(_PublicRedirects()).open(request, timeout=5) as response:
            return 200 <= response.status < 400
    except (OSError, ValueError, URLError):
        return False


def existing_artifact(value, workspace=None):
    if not isinstance(value, str) or not value.strip():
        return False
    path = Path(value).expanduser()
    if not path.is_absolute():
        if not workspace:
            return False
        path = Path(workspace) / path
    return path.is_file()


def remote_commit(evidence):
    sha = str(evidence.get("commit_sha", ""))
    remote = str(evidence.get("remote_url", ""))
    if not re.fullmatch(r"[0-9a-fA-F]{40}", sha) or not remote.startswith("https://"):
        return False
    try:
        _public_url(remote)
        result = subprocess.run(["git", "ls-remote", "--", remote], capture_output=True,
                                text=True, timeout=10, check=False)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False
    # Conservative: only commits advertised by the remote count; no local refs.
    return result.returncode == 0 and any(line.split()[0].lower() == sha.lower()
                                          for line in result.stdout.splitlines() if line.split())


def _email_evidence(task, result, evidence):
    return bool(re.fullmatch(r"[0-9a-fA-F]{12,32}", str(evidence.get("gmail_message_id", "")))
                and re.fullmatch(r"[^\s<>@]+@[^\s<>@]+\.[^\s<>@]+", str(evidence.get("sender", ""))))


def _post_evidence(task, result, evidence):
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{2,199}", str(evidence.get("publication_id", "")))) or reachable_url(evidence.get("url", ""))


def _artifact_evidence(task, result, evidence):
    artifact = evidence.get("artifact_path") or evidence.get("path")
    if not artifact and isinstance(result, str) and "\n" not in result:
        artifact = result.strip()
    valid = existing_artifact(artifact, task.workspace_path)
    if valid and task.workspace_kind == "scratch" and task.workspace_path:
        path = Path(artifact).expanduser()
        if not path.is_absolute():
            path = Path(task.workspace_path) / path
        if path.resolve().is_relative_to(Path(task.workspace_path).resolve()):
            return False
    return valid or (fields(task.body).get("task_type", "generic") == "generic"
                     and reachable_url(evidence.get("url", artifact or "")))


_VERIFIERS = {
    "send_email": (_email_evidence, "gmail_message_id (Gmail hex id) and sender email"),
    "publish_post": (_post_evidence, "publication_id or reachable url"),
    "listing": (lambda task, result, ev: reachable_url(ev.get("url", "")), "reachable profile url"),
    "code_change": (lambda task, result, ev: remote_commit(ev), "full commit_sha advertised by HTTPS remote_url"),
    "skill_change": (lambda task, result, ev: bool(ev.get("diff") and ev.get("regression")), "diff and regression evidence"),
    "review": (lambda task, result, ev: ev.get("verdict") in {"PASS", "FAIL", "NO_VERIFICADO"} and bool(ev.get("proof")), "verdict and primary proof"),
    "research": (_artifact_evidence, "existing artifact_path outside disposable scratch; attach it durably"),
    "generic": (_artifact_evidence, "existing artifact_path outside disposable scratch or reachable url"),
}


def verify_evidence(task, result):
    kind = fields(task.body).get("task_type", "generic")
    evidence = fields(result)
    if evidence.get("schema_valid") is False:
        return "schema_valid=false"
    if kind not in _VERIFIERS:
        return "unknown task_type"
    verify, required = _VERIFIERS[kind]
    return None if verify(task, result, evidence) else kind + " requires " + required


def gate_completion(conn, task_id, result, expected_run_id, force, metadata=None):
    from hermes_cli import kanban_db as kb

    from hermes_cli.kanban_tree_budget import allowed
    if not allowed(conn, task_id):
        return False
    task = kb.get_task(conn, task_id)
    if task is None or not applies(conn, task):
        return True
    if task.status not in {"running", "ready", "blocked", "review"}:
        return False
    if expected_run_id is not None and task.current_run_id != expected_run_id:
        return False
    if expected_run_id is None and not force and kb._claim_is_live(conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()):
        raise kb.LiveClaimError(task_id)
    candidate = result if result is not None else task.result
    reason = "schema_valid=false" if (metadata or {}).get("schema_valid") is False else verify_evidence(task, candidate)
    if reason is None:
        from hermes_cli.kanban_cross_review import gate
        return gate(conn, task, candidate)
    with kb.write_txn(conn):
        current = kb.get_task(conn, task_id)
        if current != task:
            return False
        conn.execute("UPDATE tasks SET status='review', result=?, claim_lock=NULL, "
                     "claim_expires=NULL, worker_pid=NULL WHERE id=?", (candidate, task_id))
        kb._end_run(conn, task_id, outcome="evidence_missing", status="review", summary=reason)
        kb._insert_comment(conn, task_id, "alfred", "Evidence gate: " + reason, int(time.time()))
        kb._append_event(conn, task_id, "evidence_rejected", {"reason": reason})
    return False


def enforce_resource_owner(conn, task_id):
    """Called within creation/assignment/claim transactions, serializing rivals."""
    from hermes_cli import kanban_db as kb

    from hermes_cli.kanban_cross_review import managed_original
    if managed_original(conn, task_id) and kb.get_task(conn, task_id).status == "review":
        return False
    task = kb.get_task(conn, task_id)
    if task is None or task.status in {"done", "archived"}:
        return True
    key = fields(task.body).get("resource_key")
    if not key or not task.assignee:
        return True
    for row in conn.execute("SELECT id, body, assignee FROM tasks WHERE status NOT IN ('done','archived') "
                            "AND id != ? ORDER BY created_at, rowid", (task_id,)):
        if row["assignee"] and row["assignee"] != task.assignee and fields(row["body"]).get("resource_key") == key:
            # A conflict-blocked card never steals ownership from its incumbent.
            rival = kb.get_task(conn, row["id"])
            if rival.status == "blocked" and rival.block_kind == "owner_conflict":
                continue
            reason = "owner_conflict: resource already owned by task " + row["id"]
            conn.execute("UPDATE tasks SET status='blocked', block_kind='owner_conflict', claim_lock=NULL, "
                         "claim_expires=NULL, worker_pid=NULL WHERE id=?", (task_id,))
            kb._end_run(conn, task_id, outcome="blocked", status="blocked", summary=reason)
            kb._insert_comment(conn, task_id, "alfred", reason, int(time.time()))
            kb._append_event(conn, task_id, "owner_conflict", {"owner_task_id": row["id"]})
            return False
    return True


def _substantive_text(value: Optional[str]) -> bool:
    return bool(value is not None and str(value).strip())


def gate_empty_completion(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: Optional[str],
    summary: Optional[str],
) -> None:
    """Refuse a completion that would leave the card with no evidence.

    Review approvals are exempt: a human vouches for the card and
    ``_REVIEW_APPROVED_NOTE`` is the documented record.
    """
    from hermes_cli.kanban_db import write_txn, _append_event, _first_line, EmptyCompletionError

    row = conn.execute(
        "SELECT status, result FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return
    if row["status"] == "review":
        return
    stored = row["result"]
    if _substantive_text(result) or _substantive_text(summary) or _substantive_text(stored):
        return
    with write_txn(conn):
        _append_event(
            conn, task_id, "completion_blocked_empty_result",
            {
                "result_preview": _first_line(result, 200) or None,
                "summary_preview": _first_line(summary, 200) or None,
            },
        )
    raise EmptyCompletionError(task_id)




def append_evidence_context(lines, conn, task):
    if applies(conn, task):
        kind = fields(task.body).get("task_type", "generic")
        lines.append("Completion evidence (" + kind + "): result must be JSON with "
                     "gmail_message_id + sender for send_email; publication_id or url for publish_post; "
                     "url for listing; commit_sha + HTTPS remote_url for code_change; "
                     "artifact_path for research; artifact_path or url for generic. "
                     "Only existing files, reachable public URLs and remote-advertised commits pass. "
                     "Keep artifacts outside disposable scratch workspaces or attach them durably. "
                     "Missing evidence stays in review, including manual approval. "
                     "Heartbeat alone is not progress; record milestones or artifacts.")


def prepare_edit(conn, task_id, body, result):
    from hermes_cli import kanban_db as kb
    task = kb.get_task(conn, task_id)
    if task is None:
        return None
    from hermes_cli.kanban_cross_review import REVIEW_TYPES
    kind = fields(task.body).get("task_type")
    if kind in REVIEW_TYPES | {"review"}:
        if body is not None and fields(body).get("task_type") != kind:
            raise ValueError("cross-review task_type cannot be removed or changed")
        if task.status in {"review", "done"} and (body is not None and body != task.body or result is not None and result != task.result):
            raise ValueError("reviewed evidence and criteria are immutable; request changes first")
    if task.status != "done":
        return task if kind in REVIEW_TYPES | {"review"} else None
    proposed = replace(task, body=body if body is not None else task.body)
    if applies(conn, proposed):
        reason = verify_evidence(proposed, result if result is not None else task.result)
        if reason:
            raise ValueError("completed task evidence cannot be invalidated: " + reason)
        return task
    return None

"""Creation-time mission and budget checks shared by CLI, tools and fan-out."""
from __future__ import annotations

import re

HEADERS = {
    "objetivo": "Objetivo", "goal": "Objetivo",
    "terminado cuando": "Terminado cuando", "done when": "Terminado cuando",
    "recibo": "Recibo", "receipt": "Recibo",
}
_HEADER = re.compile(
    r"(?im)^\s*(?:#{1,6}\s+)?(?:\*\*)?"
    r"(objetivo|goal|terminado cuando|done when|recibo|receipt)"
    r"\s*:(?:\*\*)?[ \t]*"
)


def sections(body):
    matches = list(_HEADER.finditer(body or ""))
    result = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        value = re.sub(r"(?m)^[ \t]*(?:budget_tokens|budget_seconds|task_type):[^\n]*(?:\n|$)",
                       "", body[match.end():end]).strip()
        if value:
            result[HEADERS[match[1].lower()]] = value
    return result


def missing_spec(body, defaults):
    from hermes_cli.kanban_evidence import fields

    missing = [name for name in dict.fromkeys(HEADERS.values()) if name not in sections(body)]
    values = fields(body)
    typed = defaults.get(values.get("task_type", "generic"), {})
    for key in ("budget_tokens", "budget_seconds"):
        value = values.get(key, typed.get(key, defaults.get(key)))
        try:
            valid = not isinstance(value, bool) and int(value) > 0
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            missing.append(key)
    return missing


def prepare(conn, body, creator_task_id=None):
    """Read-only preparation; rejection happens before any task is inserted."""
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli.kanban_evidence import fields
    from hermes_cli.kanban_tree_budget import root_for

    config = load_user_config_effective().get("kanban", {})
    mode = config.get("require_spec", "warn")
    if mode not in {"warn", "enforce"}:
        raise ValueError("kanban.require_spec debe ser warn o enforce")
    comments = []
    defaults = config.get("budget_defaults", {})
    if creator_task_id:
        parent = conn.execute("SELECT body FROM tasks WHERE id=?", (creator_task_id,)).fetchone()
        if parent:
            inherited = []
            own = sections(body)
            for name, value in sections(parent[0]).items():
                if name not in own:
                    body = (body or "") + f"\n\n{name}: {value}"
                    inherited.append(name)
            # Limits belong to the creation tree, not dependency edges.
            root = root_for(conn, creator_task_id)
            root_body = conn.execute("SELECT body FROM tasks WHERE id=?", (root,)).fetchone()[0]
            values = fields(root_body)
            typed = defaults.get(values.get("task_type", "generic"), {})
            defaults = {**defaults, **typed, **{k: values[k] for k in
                        ("budget_tokens", "budget_seconds") if k in values}}
            has_budgets = conn.execute("SELECT 1 FROM sqlite_master WHERE name='kanban_tree_budgets'").fetchone()
            cap = (conn.execute("SELECT budget_tokens,budget_seconds FROM kanban_tree_budgets WHERE root_id=?",
                                (root,)).fetchone() if has_budgets else None)
            if cap:
                defaults = {"budget_tokens": cap[0], "budget_seconds": cap[1]}
            if inherited:
                comments.append("especificación heredada de " + creator_task_id + ": " + ", ".join(inherited))
    missing = missing_spec(body, defaults)
    if missing:
        warning = "falta especificación: " + ", ".join(missing)
        if mode == "enforce":
            raise ValueError(warning)
        comments.append(warning)
    return body, comments


def record(conn, task_id, comments, now):
    from hermes_cli.kanban_db import _insert_comment

    for comment in comments:
        _insert_comment(conn, task_id, "kanban-spec", comment, now)

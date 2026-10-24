"""Profile validation for durable Kanban ownership (no credential/config reads)."""
from typing import Optional

from hermes_cli.profiles import list_profile_names, normalize_profile_name


MANUAL_ASSIGNEES = frozenset({"claude"})


def is_manual_assignee(name: Optional[str]) -> bool:
    """Interactive clients pull these lanes; Hermes never launches them."""
    return bool(name and name.strip().lower() in MANUAL_ASSIGNEES)


def canonical_assignee(assignee: Optional[str]) -> Optional[str]:
    """Normalize filters as well as writes, including Alfred's self alias."""
    if assignee is None:
        return None
    name = normalize_profile_name(assignee)
    return "default" if name == "alfred" else name


def validated_assignee(assignee: Optional[str]) -> Optional[str]:
    """Reject unknown owners before any task/claim mutation; None unassigns."""
    if assignee is None:
        return None
    profiles = set(list_profile_names()) | MANUAL_ASSIGNEES
    valid = f"{', '.join(sorted(profiles))} (alfred = default)"
    try:
        name = canonical_assignee(assignee)
    except ValueError as exc:
        raise ValueError(f"{exc}. Valid profiles: {valid}") from None
    if name not in profiles:
        raise ValueError(
            f"Unknown assignee {assignee!r}. Valid profiles: "
            f"{valid}. "
            "Choose an existing profile, not a skill name."
        )
    return name


def known_assignees(conn) -> list[dict]:
    """Discover on-disk, historical and manual owners before their first task."""
    from hermes_cli import kanban_db as kb
    on_disk = set(kb.list_profiles_on_disk())
    counts = kb._counts_by_assignee(conn)
    return [
        {"name": name, "on_disk": name in on_disk, "counts": counts.get(name, {})}
        for name in sorted(on_disk | set(counts) | MANUAL_ASSIGNEES)
    ]

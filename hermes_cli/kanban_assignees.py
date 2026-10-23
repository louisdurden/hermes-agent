"""Profile validation for durable Kanban ownership (no credential/config reads)."""
from typing import Optional

from hermes_cli.profiles import list_profile_names, normalize_profile_name


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
    profiles = list_profile_names()
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

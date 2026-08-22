"""Graph-of-Agents one-turn configuration helpers.

GoA is deliberately opt-in.  The marker is transport-safe text used by the
CLI/TUI/gateway paths that cannot carry a second structured turn argument.
"""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from typing import Any

GOA_MARKER_PREFIX = "__HERMES_GOA_TURN_V1__"
DEFAULT_GOA_PROFILES = (
    "fleetauthority", "fleetcontent", "fleetestetica", "fleetfabric",
    "fleetliterature", "fleetoutreach", "fleetpremium", "fleetseo",
)
DEFAULT_GOA_META = {
    "provider": "openai-codex", "model": "gpt-5.6-sol", "reasoning_effort": "medium",
}


def normalize_goa_config(raw: Any) -> dict[str, Any]:
    """Tolerantly normalize the small, explicitly one-turn GoA contract."""
    raw = raw if isinstance(raw, dict) else {}
    profiles = raw.get("profiles")
    if not isinstance(profiles, list):
        profiles = list(DEFAULT_GOA_PROFILES)
    profiles = [str(profile).strip() for profile in profiles if str(profile).strip()]
    allowed = tuple(profile for profile in profiles if profile in DEFAULT_GOA_PROFILES)
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "top_k": min(3, max(1, int(raw.get("top_k", 3) or 3))),
        "rounds": 1,
        "edge_threshold": float(raw.get("edge_threshold", 0.0) or 0.0),
        "profiles": allowed or DEFAULT_GOA_PROFILES,
        "meta": {**deepcopy(DEFAULT_GOA_META), **{k: meta[k] for k in DEFAULT_GOA_META if k in meta}},
        "token_budget": 7000,
        "invocation_budget": 12,
    }


def encode_goa_turn(prompt: str, config: Any = None) -> str:
    payload = {"prompt": str(prompt or ""), "config": normalize_goa_config(config or {})}
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    return f"{GOA_MARKER_PREFIX}{encoded}"


def decode_goa_turn(message: Any) -> tuple[str, dict[str, Any] | None]:
    if not isinstance(message, str) or not message.startswith(GOA_MARKER_PREFIX):
        return message, None
    try:
        payload = json.loads(base64.urlsafe_b64decode(message[len(GOA_MARKER_PREFIX):].strip()).decode())
    except Exception:
        return message, None
    return str(payload.get("prompt") or ""), normalize_goa_config(payload.get("config") or {})


def build_goa_turn_prompt(user_prompt: str, config: Any = None) -> str:
    return encode_goa_turn(user_prompt, config)

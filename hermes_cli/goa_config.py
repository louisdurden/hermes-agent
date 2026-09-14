"""Fail-closed Graph-of-Agents configuration helpers."""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from typing import Any

GOA_MARKER_PREFIX = "__HERMES_GOA_TURN_V1__"
DEFAULT_GOA_PROFILES = (
    "fleetauthority",
    "fleetcontent",
    "fleetestetica",
    "fleetfabric",
    "fleetliterature",
    "fleetoutreach",
    "fleetpremium",
    "fleetseo",
)
DEFAULT_GOA_META = {
    "provider": "openai-codex",
    "model": "gpt-5.6-sol",
    "reasoning_effort": "medium",
}


def normalize_goa_config(raw: Any) -> dict[str, Any]:
    """Normalize the bounded one-turn contract; only literal true enables it."""
    raw = raw if isinstance(raw, dict) else {}
    profiles = raw.get("profiles")
    if not isinstance(profiles, list):
        profiles = list(DEFAULT_GOA_PROFILES)
    allowed = tuple(
        profile
        for value in profiles
        if (profile := str(value).strip()) in DEFAULT_GOA_PROFILES
    )
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    try:
        top_k = int(raw.get("top_k", 3) or 3)
    except (TypeError, ValueError):
        top_k = 3
    try:
        edge_threshold = float(raw.get("edge_threshold", 0.0) or 0.0)
    except (TypeError, ValueError):
        edge_threshold = 0.0
    return {
        "enabled": raw.get("enabled") is True,
        "top_k": min(3, max(1, top_k)),
        "rounds": 1,
        "edge_threshold": edge_threshold,
        "profiles": allowed or DEFAULT_GOA_PROFILES,
        "meta": {
            **deepcopy(DEFAULT_GOA_META),
            **{key: meta[key] for key in DEFAULT_GOA_META if key in meta},
        },
        "token_budget": 7000,
        "invocation_budget": 12,
    }


def get_goa_config(config: Any) -> dict[str, Any]:
    root = config if isinstance(config, dict) else {}
    return normalize_goa_config(root.get("goa"))


def is_goa_enabled(config: Any) -> bool:
    return get_goa_config(config)["enabled"] is True


def encode_goa_turn(prompt: str, config: Any = None) -> str:
    payload = {"prompt": str(prompt or ""), "config": normalize_goa_config(config)}
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode()
    return f"{GOA_MARKER_PREFIX}{encoded}"


def decode_goa_turn(message: Any) -> tuple[Any, dict[str, Any] | None]:
    if not isinstance(message, str) or not message.startswith(GOA_MARKER_PREFIX):
        return message, None
    try:
        encoded = message[len(GOA_MARKER_PREFIX) :].strip()
        payload = json.loads(base64.urlsafe_b64decode(encoded).decode())
        if not isinstance(payload, dict):
            return message, None
    except Exception:
        return message, None
    return str(payload.get("prompt") or ""), normalize_goa_config(payload.get("config"))


def build_goa_turn_prompt(user_prompt: str, config: Any = None) -> str:
    return encode_goa_turn(user_prompt, config)


def goa_usage() -> str:
    return "Usage: /goa <prompt>  (requires goa.enabled: true)"

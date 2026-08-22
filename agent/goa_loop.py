"""Opt-in Graph-of-Agents profile coordination for one Hermes turn.

The twelve bounded, tool-free auxiliary calls are: selection; three profile
answers; three peer rankings; four bidirectional refinements; and pooling.
Profiles remain workers, while the neutral meta model alone selects/pools.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agent.auxiliary_client import call_llm

logger = logging.getLogger(__name__)
_STAGE_TOKENS = {"selection": 300, "answer": 900, "ranking": 200, "refinement": 600, "pooling": 1000}


@dataclass(frozen=True)
class AgentCard:
    agent_id: str
    domain: str
    specializations: tuple[str, ...]


CARDS = (
    AgentCard("fleetauthority", "personal authority and distribution", ("positioning", "distribution")),
    AgentCard("fleetcontent", "clinical social content", ("clinical content", "quality assurance")),
    AgentCard("fleetestetica", "Estetica-AI engineering", ("engineering", "canaries")),
    AgentCard("fleetfabric", "Alfred platform operations", ("jobs", "scheduler", "tests")),
    AgentCard("fleetliterature", "scientific literature", ("evidence", "citations")),
    AgentCard("fleetoutreach", "Forma Agent outreach preparation", ("discovery", "qualification")),
    AgentCard("fleetpremium", "premium product quality", ("audit", "rubric", "quality gate")),
    AgentCard("fleetseo", "SEO and B2B audit", ("analytics", "SEO", "site checkpoints")),
)


def _text(response: Any) -> str:
    try:
        return str(response.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _call(meta: dict[str, Any], prompt: str, *, stage: str) -> str:
    return _text(call_llm(
        task=f"goa_{stage}", provider=meta["provider"], model=meta["model"],
        messages=[{"role": "system", "content": "You are a tool-free GoA coordinator. Return only the requested concise analysis."}, {"role": "user", "content": prompt}],
        max_tokens=_STAGE_TOKENS[stage], reasoning_config={"effort": meta.get("reasoning_effort", "medium")}, tools=None,
    ))


def _parse_selection(text: str, permitted: Sequence[str], top_k: int) -> list[str]:
    try:
        parsed = json.loads(text)
        values = parsed.get("selected", parsed) if isinstance(parsed, dict) else parsed
        selected = [str(value) for value in values if str(value) in permitted]
    except Exception:
        selected = []
    return list(dict.fromkeys(selected))[:top_k]


def _profile_session(profile: str) -> tuple[dict[str, Any], str]:
    """Resolve a profile's own Hermes model and public session identity.

    This is deliberately a read-only projection: the GoA node never switches
    the parent process's ``HERMES_HOME`` and therefore cannot leak one
    profile's config into another node or into the acting conversation.
    """
    fallback = {"provider": "openai-codex", "model": "gpt-5.6-sol", "reasoning_effort": "medium"}
    try:
        from hermes_cli import profiles

        profile_dir = profiles.get_profile_dir(profile)
        model, provider = profiles._read_config_model(profile_dir)
        soul_path = profile_dir / "SOUL.md"
        soul = soul_path.read_text(encoding="utf-8")[:8000] if soul_path.is_file() else ""
        return {**fallback, "provider": provider or fallback["provider"], "model": model or fallback["model"]}, soul
    except Exception:
        logger.warning("GoA profile session %s could not be resolved; using its bounded fallback runtime", profile)
        return fallback, ""


def _profile_call(profile: str, question: str, *, stage: str, own: str = "", references: Mapping[str, str] | None = None) -> str:
    refs = "\n".join(f"{name}: {value}" for name, value in (references or {}).items())
    runtime, soul = _profile_session(profile)
    prompt = f"You are Hermes profile {profile}. Work within its specialty. No tools.\nProfile session identity:\n{soul}\nTask: {question}\nOwn draft: {own}\nPeer material:\n{refs}\nReturn a concise {stage}."
    return _call(runtime, prompt, stage=stage)


def aggregate_goa_context(*, user_prompt: str, config: dict[str, Any]) -> str:
    """Run exactly one bounded GoA graph and return private acting-agent guidance."""
    meta = config["meta"]
    permitted = [card.agent_id for card in CARDS if card.agent_id in config["profiles"]]
    card_text = "\n".join(f"{card.agent_id}: {card.domain}; {', '.join(card.specializations)}" for card in CARDS if card.agent_id in permitted)
    selection = _call(meta, f"Select exactly {config['top_k']} relevant profile ids as JSON {{\"selected\":[...]}}. Never select fleetfabric.\nTask: {user_prompt}\nCards:\n{card_text}", stage="selection")
    selected = [profile for profile in _parse_selection(selection, permitted, config["top_k"]) if profile != "fleetfabric"]
    if len(selected) != config["top_k"]:
        selected = [profile for profile in permitted if profile != "fleetfabric"][:config["top_k"]]
    initial: dict[str, str] = {}
    failed: list[str] = []
    for profile in selected:
        try:
            initial[profile] = _profile_call(profile, user_prompt, stage="answer")
        except Exception:
            failed.append(profile)
    live = [profile for profile in selected if initial.get(profile)]
    if len(live) < 2:
        return "[Graph-of-Agents degraded: fewer than two profile nodes responded; proceed without GoA guidance.]"
    scores = {profile: 0.0 for profile in live}
    for judge in live:
        candidates = {profile: initial[profile] for profile in live if profile != judge}
        try:
            raw = _profile_call(judge, user_prompt, stage="ranking", references=candidates)
            parsed = json.loads(raw)
            ranking = parsed.get("scores", parsed) if isinstance(parsed, dict) else {}
            for profile in candidates:
                scores[profile] += max(0.0, float(ranking.get(profile, 0.0)))
        except Exception:
            failed.append(judge)
    ranked = sorted(live, key=lambda profile: (-scores[profile], profile))
    source_edges = {source: tuple(ranked[index + 1:]) for index, source in enumerate(ranked[:-1])}
    target_edges: dict[str, list[str]] = {profile: [] for profile in live}
    for source, targets in source_edges.items():
        for target in targets:
            target_edges[target].append(source)
    refined = dict(initial)
    for target, sources in target_edges.items():
        if sources:
            try:
                refined[target] = _profile_call(target, user_prompt, stage="refinement", own=refined[target], references={source: refined[source] for source in sources})
            except Exception:
                failed.append(target)
    for source, targets in source_edges.items():
        if targets:
            try:
                refined[source] = _profile_call(source, user_prompt, stage="refinement", own=refined[source], references={target: refined[target] for target in targets})
            except Exception:
                failed.append(source)
    pooled = _call(meta, f"Pool these GoA profile outputs into private guidance. Preserve provenance, disagreements, and limits.\nTask: {user_prompt}\nOutputs:\n{json.dumps(refined, ensure_ascii=False)}", stage="pooling")
    degraded = f" Failed nodes: {', '.join(sorted(set(failed)))}." if failed else ""
    return f"[Graph-of-Agents guidance; selected: {', '.join(selected)}; 12-call/7000-token ceiling.{degraded}]\n{pooled}"

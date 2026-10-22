"""Owner-only deterministic fleet status, independent of model turns."""

from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path

from gateway.config import Platform
from gateway.platforms._shared import platform_gate_env


def compact_fleet_status(text: str) -> str:
    """Reserve space for every operational priority before optional detail."""
    sections: dict[str, list[str]] = {}
    section = "header"
    for line in text.splitlines():
        if not line.strip():
            continue
        if not line.startswith(" "):
            section = line.split(":", 1)[0]
        sections.setdefault(section, []).append(line)

    agents = sections.get("Kanban alfred-fleet", [])
    urgent_agents = [line for line in agents[1:] if re.search(r"(?:blocked|stalled)=[1-9]", line)]
    jobs = sections.get("Cron", [])
    failing_jobs = [line for line in jobs[1:] if (
        re.search(r"fallos 24 h: [1-9]", line)
        or re.search(r"\[(?:error|failed|failure)\]", line, re.IGNORECASE)
        or "incidentes abiertos:" in line
    )]
    commercial = sections.get("Comercial", [])
    today = [line for line in commercial[1:] if "enviados 7 d:" not in line]
    groups = [
        (sections.get("Gateway", ["Gateway: sin datos"]), 400),
        (["Agentes bloqueados/estancados:", *(urgent_agents or agents[1:] or ["  sin datos"])], 900),
        (["Cron:", *(failing_jobs or jobs[1:] or ["  sin datos"])], 900),
        (["Comercial:", *(today or ["  sin datos"])], 550),
        (sections.get("Alertas", ["Alertas: sin datos"]), 1000),
    ]
    result = []
    for lines, budget in groups:
        block = "\n".join(lines)
        encoded = block.encode("utf-16-le")
        result.append(block if len(encoded) <= budget * 2 else
                      encoded[:(budget - 2) * 2].decode("utf-16-le", errors="ignore") + " …")
    return "\n\n".join(result)


def _read_fleet_status() -> str:
    script = Path.home() / "alfred" / "scripts" / "fleet_status.py"
    try:
        result = subprocess.run(
            ["python3", str(script)], capture_output=True, text=True,
            timeout=25, check=False,
        )
    except subprocess.TimeoutExpired:
        return "Estado de flota: se agotó el tiempo de consulta (25 s)."
    except OSError:
        return "Estado de flota: no se pudo ejecutar la consulta local."
    if result.returncode != 0:
        return "Estado de flota: la consulta local falló."
    if not result.stdout.strip():
        return "Estado de flota: la consulta local no devolvió datos."
    return compact_fleet_status(result.stdout)


class GatewayEstadoCommandsMixin:
    async def _handle_estado_command(self, event) -> str:
        source = event.source
        if source is None or source.platform != Platform.TELEGRAM or source.chat_type != "dm":
            return "El comando /estado está restringido al chat privado autorizado de Telegram."
        adapter = self._intake_adapter_for(source)
        home = adapter.config.home_channel if adapter is not None else None
        owner = str(home.chat_id) if home else ""
        allowed = {item.strip() for item in platform_gate_env("TELEGRAM_ALLOWED_USERS").split(",")}
        if not (owner.isdigit() and owner in allowed
                and str(source.chat_id) == owner and str(source.user_id) == owner
                and (event.user_id is None or str(event.user_id) == owner)):
            return "El comando /estado está restringido al chat privado autorizado de Telegram."
        return await asyncio.to_thread(_read_fleet_status)

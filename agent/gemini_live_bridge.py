"""Gemini Live bridge — segundo cerebro de Alfred, solo para voz en vivo. FX-142.

Por qué existe: el modo de voz actual del Desktop es una cascada (STT local ->
texto -> Codex/gpt-5.6-sol -> texto -> TTS) — cada paso agrega latencia y
rompe la interrupción natural. Gemini Live es un modelo de voz-a-voz nativo:
sin transcripción intermedia, con barge-in real manejado por el propio
servidor. Es un camino PARALELO, no un reemplazo: Telegram y el resto de
Alfred (62 tools, gbrain, research) siguen en Codex exactamente igual.

Diseño: el Desktop nunca habla directo con la API de Google (la key nunca
sale de este proceso) -- se conecta a un WebSocket local acá, y este módulo
hace de puente hacia `wss://generativelanguage.googleapis.com/...` real,
reinyectando la key servidor-side. Maneja la reconexión por límite de
duración de sesión (el cierre "GoAway" que confirmamos en el prototipo)
de forma transparente para el cliente.

Alcance de ESTA primera integración (verificado, no una promesa de más):
persona real (SOUL.md) + voz elegida (Autonoe) + audio bidireccional +
reconexión automática. Tool-calling (brainTools/gbrain) queda FUERA de este
primer corte -- una sesión de Gemini Live hoy no tiene ninguna herramienta de
Alfred; es conversación pura. Ver ledger FX-142 para el porqué de ese límite
y el plan de la siguiente vuelta.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from datetime import datetime, timezone
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from agent.curated_memory import search_curated_memory
from agent.gemini_schema import sanitize_gemini_tool_parameters

logger = logging.getLogger(__name__)

GEMINI_LIVE_MODEL = "models/gemini-3.1-flash-live-preview"
GEMINI_LIVE_VOICE = "Autonoe"
GEMINI_WS_URL_TEMPLATE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={key}"
)

# Persona condensada de SOUL.md — se relee en vivo en cada sesión nueva (no un
# snapshot congelado en el código) para que un cambio a SOUL.md se refleje sin
# tocar este archivo. Fallback si SOUL.md no está disponible.
_FALLBACK_PERSONA = (
    "Sos Alfred, el mayordomo digital del Dr. Rodrigo Hamuy. Hablas en espanol "
    "neutro latinoamericano, SIN voseo (usa siempre tu). Tu voz es femenina y "
    "calida. Respuestas breves y conversacionales, como si hablaras en vivo. "
    "Tenes acceso limitado a memoria, agenda y recordatorios; usa esas herramientas "
    "solo ante pedidos explícitos de Rodrigo."
)

_ALFRED_HOME = Path(os.environ.get("ALFRED_HOME", str(Path.home() / "alfred")))
_GBRAIN_BIN = os.environ.get("SPINE_GBRAIN_BIN", str(Path.home() / ".bun" / "bin" / "gbrain"))
_CURATED_MEMORY_BUNDLE = Path(
    os.environ.get("ALFRED_CLOUD_MEMORY_BUNDLE", "/app/.hermes/cloud-memory/curated_gbrain.json")
)
_CALENDAR_PULSE = _ALFRED_HOME / "scripts" / "calendar-pulse.cjs"
_REMINDERS_MODULE = _ALFRED_HOME / "telegram-bot" / "dist" / "reminders.js"
_SENSITIVE_DATA_GATE = _ALFRED_HOME / "security" / "sensitive-data.cjs"
_MEMORY_PERSISTENCE_GATE = _ALFRED_HOME / "security" / "memory-persistence-gate.cjs"

# Las especificaciones se conservan con la forma OpenAI que ya usa Hermes y se
# sanitizan justo antes de enviarlas. Así no nace otro dialecto de schemas para
# Live distinto del adaptador Gemini nativo.
_LIVE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "buscar_memoria",
        "description": "Busca en la memoria permanente de Rodrigo (gbrain).",
        "parameters": {
            "type": "object",
            "properties": {"consulta": {"type": "string", "description": "Término de búsqueda."}},
            "required": ["consulta"],
        },
    },
    {
        "name": "guardar_en_memoria",
        "description": "Guarda un hecho o decisión no sensible en la memoria permanente. Nunca uses para PHI ni secretos.",
        "parameters": {
            "type": "object",
            "properties": {
                "hecho": {"type": "string", "description": "Hecho o decisión, máximo 500 caracteres; sin PHI."},
                "categoria": {"type": "string", "description": "personal, proyectos, decisiones, preferencias, recordatorio u otro."},
            },
            "required": ["hecho"],
        },
    },
    {
        "name": "agenda_hoy",
        "description": "Consulta la agenda de Google Calendar de Rodrigo para las próximas 48 horas.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "crear_recordatorio",
        "description": "Crea un recordatorio que Alfred enviará por Telegram a la hora indicada. Úsala solamente ante un pedido explícito de Rodrigo.",
        "parameters": {
            "type": "object",
            "properties": {
                "texto": {"type": "string", "description": "Texto del recordatorio."},
                "cuando_iso": {"type": "string", "description": "Fecha y hora ISO 8601 en zona Asunción (UTC-3)."},
            },
            "required": ["texto", "cuando_iso"],
        },
    },
]


async def _run_local_command(*args: str, input_text: str = "", timeout: float) -> tuple[int, str]:
    """Ejecuta un backend local compartido sin exponer stderr al modelo."""
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(input_text.encode("utf-8")), timeout)
        return process.returncode or 0, stdout.decode("utf-8", errors="replace")
    except (OSError, asyncio.TimeoutError):
        if process is not None and process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
                await process.wait()
        return 1, ""


async def _run_node_json(script: str, payload: dict[str, Any], *, timeout: float = 15.0) -> Optional[dict[str, Any]]:
    code, stdout = await _run_local_command("node", "-e", script, input_text=json.dumps(payload), timeout=timeout)
    if code != 0 or not stdout:
        return None
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


async def _guard_live_tool(name: str, args: dict[str, Any]) -> str:
    """Reusa literalmente los gates de secretos/PHI del Telegram brain."""
    script = f"""
const {{ containsSecret }} = require({json.dumps(str(_SENSITIVE_DATA_GATE))});
const {{ inspectMemoryPersistence }} = require({json.dumps(str(_MEMORY_PERSISTENCE_GATE))});
let payload;
try {{ payload = JSON.parse(require('fs').readFileSync(0, 'utf8')); }} catch {{ process.stdout.write(JSON.stringify({{reason:'unserializable'}})); process.exit(0); }}
try {{
  const serialized = JSON.stringify({{name: payload.name, input: payload.args}});
  if (typeof serialized !== 'string') throw new Error('unserializable');
  if (containsSecret(serialized)) {{ process.stdout.write(JSON.stringify({{reason:'secret'}})); process.exit(0); }}
  if (payload.name === 'guardar_en_memoria') {{
    const persistence = inspectMemoryPersistence(serialized, {{secretPolicy:'block'}});
    if (!persistence.allowed && persistence.action === 'block_phi') {{ process.stdout.write(JSON.stringify({{reason:'phi'}})); process.exit(0); }}
  }}
  process.stdout.write(JSON.stringify({{reason:'allow'}}));
}} catch {{ process.stdout.write(JSON.stringify({{reason:'unserializable'}})); }}
"""
    result = await _run_node_json(script, {"name": name, "args": args})
    return str((result or {}).get("reason") or "unserializable")


async def _sanitize_live_tool_result(result: str) -> str:
    """Mismo redactor final que Telegram, aun si falló un backend local."""
    script = f"""
const {{ redactSensitiveText }} = require({json.dumps(str(_SENSITIVE_DATA_GATE))});
let payload = {{}}; try {{ payload = JSON.parse(require('fs').readFileSync(0, 'utf8')); }} catch {{}}
process.stdout.write(JSON.stringify({{text: redactSensitiveText(String(payload.result ?? '')).text}}));
"""
    sanitized = await _run_node_json(script, {"result": result})
    return str((sanitized or {}).get("text") or "(la herramienta no devolvió un resultado)")


async def _execute_live_tool(name: str, args: Any) -> str:
    """Adaptador mínimo a los backends reales ya usados por Telegram."""
    if not isinstance(args, dict):
        return "(argumentos inválidos para la herramienta)"

    guard = await _guard_live_tool(name, args)
    if guard == "secret":
        return "(herramienta bloqueada: detecté una credencial; no se ejecutó)"
    if guard == "phi":
        return "(herramienta bloqueada: detecté datos identificables de paciente; no se enviaron ni guardaron)"
    if guard != "allow":
        return "(herramienta bloqueada: argumentos no seguros)"

    result: str
    if name == "buscar_memoria":
        query = str(args.get("consulta", ""))
        # Railway cannot reach the Mac's localhost Postgres.  Its curated
        # snapshot is intentionally read-only and is never allowed to fall
        # back to a local gbrain CLI there.
        if os.environ.get("RAILWAY_ENVIRONMENT") or _CURATED_MEMORY_BUNDLE.exists():
            result = search_curated_memory(_CURATED_MEMORY_BUNDLE, query)
        else:
            code, stdout = await _run_local_command(
                _GBRAIN_BIN, "search", query, "--limit", "6", "--mode", "conservative", timeout=15.0
            )
            if code != 0 or not stdout:
                result = "(la memoria no respondió)"
            else:
                blocks: list[list[str]] = []
                for line in stdout.splitlines():
                    if re.match(r"^\[\d\.\d+\]", line):
                        blocks.append([line])
                    elif blocks and line.strip():
                        blocks[-1].append(line)
                keep = [block for block in blocks if not re.search(r"\]\s*(skills|daily)/", block[0])][:5]
                result = "\n\n".join("\n".join(block) for block in keep)[:1800] if keep else "(sin resultados en la memoria)"
    elif name == "guardar_en_memoria":
        hecho = str(args.get("hecho", ""))[:500].strip()
        if not hecho:
            result = "(el hecho no puede estar vacío)"
        else:
            category = re.sub(r"[^a-z]", "", str(args.get("categoria", "personal"))) or "personal"
            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%Hh%M")
            title = hecho[:80].replace('"', '\\"')
            content = f'---\ntitle: "{title}"\ntags: [{category}, telegram]\n---\n\n{hecho}'
            # `gbrain put <slug>` toma el contenido por STDIN, no por un flag
            # `--content` (verificado corriendo el CLI real: con `--content` como
            # argumento posicional el comando sale con exit 0 pero no crea nada
            # -- fallo silencioso. El contenido va como input_text.
            code, _stdout = await _run_local_command(
                _GBRAIN_BIN, "put", f"facts/{category}/{timestamp}", input_text=content, timeout=15.0
            )
            result = "Guardado en memoria ✓" if code == 0 else "(no pude guardar en la memoria)"
    elif name == "agenda_hoy":
        code, stdout = await _run_local_command("node", str(_CALENDAR_PULSE), timeout=30.0)
        try:
            events = json.loads(stdout).get("eventos") if code == 0 else None
        except (json.JSONDecodeError, AttributeError):
            events = None
        if not isinstance(events, list):
            result = "Sin cuentas de calendario configuradas todavía (falta la autorización de Rodrigo)."
        elif not events:
            result = "Agenda libre: sin eventos en las próximas 48 horas."
        else:
            def _format_event(event: dict) -> str:
                source = event.get("calendario") or event.get("cuenta")
                suffix = f" [{source}]" if source else ""
                return (
                    f"{event.get('inicio', '?')} → {event.get('fin', '?')} · "
                    f"{event.get('titulo', '(sin título)')}{suffix}"
                )

            result = "\n".join(
                _format_event(event) for event in events if isinstance(event, dict)
            )[:1800]
    elif name == "crear_recordatorio":
        script = f"""
const {{ queueReminder, formatReminderDue, REMINDERS_PATH }} = require({json.dumps(str(_REMINDERS_MODULE))});
let payload = {{}}; try {{ payload = JSON.parse(require('fs').readFileSync(0, 'utf8')); }} catch {{}}
const texto = String(payload.texto ?? '').trim();
const iso = String(payload.cuando_iso ?? '');
const dated = /(?:Z|[+-]\\d{{2}}:?\\d{{2}})$/i.test(iso) ? iso : `${{iso}}-03:00`;
const dueMs = new Date(dated).getTime();
if (!texto || texto.length > 2000) {{ process.stdout.write(JSON.stringify({{text:'(el texto del recordatorio está vacío o es demasiado largo)'}})); }}
else if (Number.isNaN(dueMs) || dueMs <= Date.now()) {{ process.stdout.write(JSON.stringify({{text:'(fecha/hora inválida o ya pasada — indicá un momento futuro)'}})); }}
else {{ try {{ queueReminder({{text: texto, dueMs}}, Date.now(), REMINDERS_PATH); process.stdout.write(JSON.stringify({{text:`Recordatorio creado ✓ — te aviso el ${{formatReminderDue(dueMs)}} (Asunción)`}})); }} catch {{ process.stdout.write(JSON.stringify({{text:'(no pude crear el recordatorio)'}})); }} }}
"""
        reminder = await _run_node_json(script, {"texto": args.get("texto"), "cuando_iso": args.get("cuando_iso")})
        result = str((reminder or {}).get("text") or "(no pude crear el recordatorio)")
    else:
        result = "(herramienta desconocida)"

    return await _sanitize_live_tool_result(result)


def _load_persona_instruction() -> str:
    """Lee SOUL.md real; si no existe o falla, usa el fallback embebido."""
    soul_path = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / "SOUL.md"
    try:
        text = soul_path.read_text(encoding="utf-8").strip()
        if text:
            return (
                text
                + "\n\nNOTA DE MODO DE VOZ EN VIVO: estas hablando por un canal de "
                "voz-a-voz nativo (Gemini Live), no por el pipeline de texto habitual. "
                "En este modo tenes acceso limitado a buscar y guardar memoria, consultar "
                "la agenda y crear recordatorios. Guarda memoria o crea un recordatorio "
                "solo ante un pedido explícito de Rodrigo. No afirmes tener otras herramientas."
            )
    except OSError:
        pass
    return _FALLBACK_PERSONA


class GeminiLiveSession:
    """Un puente activo cliente(Desktop) <-> Gemini Live, con reconexión.

    Uso: `session = GeminiLiveSession(api_key); await session.run(client_ws)`.
    `run` no retorna hasta que el cliente cierra la conexión; reconecta contra
    Gemini de forma transparente cuando la sesión llega a su límite de
    duración (cierre 1008 con GoAway), sin que el cliente note el corte.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._persona = _load_persona_instruction()
        self._resumption_handle: Optional[str] = None

    @staticmethod
    def _live_tool_declarations() -> list[dict[str, Any]]:
        """Convierte las cuatro schemas Hermes al subset aceptado por Gemini."""
        declarations: list[dict[str, Any]] = []
        for tool in _LIVE_TOOLS:
            declarations.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": sanitize_gemini_tool_parameters(tool["parameters"]),
                }
            )
        return declarations

    async def _reply_to_tool_call(self, upstream, tool_call: Any) -> None:
        """Ejecuta el lote pedido por Live y lo devuelve al mismo WebSocket upstream."""
        calls = tool_call.get("functionCalls") if isinstance(tool_call, dict) else None
        if not isinstance(calls, list):
            return

        responses: list[dict[str, Any]] = []
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            call_id = call.get("id")
            if not isinstance(name, str) or not name or not isinstance(call_id, str) or not call_id:
                continue
            result = await _execute_live_tool(name, call.get("args", {}))
            # Live correlaciona cada respuesta con su llamada por id. No se
            # reenvía este mensaje al Desktop: es el round-trip interno Gemini
            # <-> bridge documentado para BidiGenerateContent:
            # https://ai.google.dev/gemini-api/docs/live-api/get-started-websocket
            responses.append({"id": call_id, "name": name, "response": {"result": result}})

        if responses:
            await upstream.send(json.dumps({"toolResponse": {"functionResponses": responses}}))

    async def run(self, client_ws) -> None:
        """`client_ws` es el WebSocket del Desktop (servidor FastAPI)."""
        from starlette.websockets import WebSocketDisconnect

        client_closed = asyncio.Event()

        async def pump_client_to_upstream(upstream, stop_event: asyncio.Event) -> None:
            try:
                while not stop_event.is_set():
                    msg = await client_ws.receive_text()
                    await upstream.send(msg)
            except WebSocketDisconnect:
                client_closed.set()
            except Exception:  # noqa: BLE001 - la reconexión de arriba decide qué hacer
                logger.debug("gemini_live: client->upstream pump terminó", exc_info=True)
            finally:
                stop_event.set()

        while not client_closed.is_set():
            reconnect_needed = await self._run_one_upstream_session(
                client_ws, pump_client_to_upstream, client_closed
            )
            if not reconnect_needed:
                break
            print("[gemini_live] sesión de Gemini vencida por duración, reconectando…", flush=True)

    async def _run_one_upstream_session(self, client_ws, pump_fn, client_closed: asyncio.Event) -> bool:
        """Una conexión real a Gemini Live. -> True si hay que reconectar (GoAway)."""
        import websockets

        url = GEMINI_WS_URL_TEMPLATE.format(key=self._api_key)
        stop_event = asyncio.Event()
        goaway = False

        print(f"[gemini_live] conectando a upstream… resumption_handle={self._resumption_handle}", flush=True)
        try:
            # open_timeout default de la librería (10s) resultó corto en una
            # prueba real (TimeoutError: timed out during opening handshake) --
            # 30s da margen sin que un fallo real de red tarde una eternidad.
            async with websockets.connect(url, max_size=None, open_timeout=30) as upstream:
                print("[gemini_live] upstream conectado, enviando setup…", flush=True)
                setup: dict = {
                    "model": GEMINI_LIVE_MODEL,
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": GEMINI_LIVE_VOICE}}
                        },
                    },
                    "systemInstruction": {"parts": [{"text": self._persona}]},
                    "tools": [{"functionDeclarations": self._live_tool_declarations()}],
                    # Habilita que Gemini nos mande sessionResumptionUpdate en
                    # cada turno; guardamos el handle más reciente para poder
                    # retomar la MISMA sesión lógica en la próxima reconexión
                    # (si no, cada reconexión arranca una charla nueva sin
                    # contexto -- lo que se sentía como "dejó de responder").
                    "sessionResumption": (
                        {"handle": self._resumption_handle} if self._resumption_handle else {}
                    ),
                }
                await upstream.send(json.dumps({"setup": setup}))

                pump_task = asyncio.create_task(pump_fn(upstream, stop_event))

                try:
                    async for raw in upstream:
                        # El WS de Gemini Live manda sus mensajes JSON (incluido el
                        # audio, como base64 dentro del JSON) en frames BINARIOS, no
                        # de texto -- `websockets` los entrega como `bytes`. Hay que
                        # decodificarlos como UTF-8 y tratarlos igual que texto; el
                        # bug real (FX-142 bugfix): reenviarlos crudos como binario
                        # dejaba al cliente sin poder parsear el JSON -> cero audio.
                        # Nada de logging por-mensaje acá: con audio a este ritmo un
                        # print(flush=True) por chunk introduce cortes audibles reales.
                        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                        try:
                            parsed = json.loads(text)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            await client_ws.send_text(text)
                            continue

                        handle = parsed.get("sessionResumptionUpdate", {}).get("newHandle")
                        if handle:
                            self._resumption_handle = handle

                        if "toolCall" in parsed:
                            await self._reply_to_tool_call(upstream, parsed["toolCall"])
                            continue

                        await client_ws.send_text(text)

                        if "goAway" in parsed:
                            goaway = True
                            print(f"[gemini_live] goAway recibido: {parsed.get('goAway')}", flush=True)
                except Exception:  # noqa: BLE001
                    logger.debug("gemini_live: upstream->client pump terminó", exc_info=True)
                finally:
                    stop_event.set()
                    pump_task.cancel()
                    # CancelledError hereda de BaseException desde Python 3.8 --
                    # suppress(Exception) NO lo atrapa. Sin esto, cada reconexión
                    # (p.ej. por el límite de duración de sesión de Gemini, ~2min)
                    # tira una excepción sin manejar que aborta todo el puente en
                    # vez de reconectar. Bug real confirmado por traceback (FX-142).
                    with contextlib.suppress(asyncio.CancelledError):
                        await pump_task
        except Exception as exc:  # noqa: BLE001
            print(f"[gemini_live] conexión upstream falló: {exc!r}", flush=True)
            with contextlib.suppress(Exception):
                await client_ws.send_text(json.dumps({"type": "bridge_error", "error": str(exc)}))
            return False

        will_reconnect = goaway and not client_closed.is_set()
        print(
            f"[gemini_live] sesión terminó (goaway={goaway}, "
            f"client_closed={client_closed.is_set()}) -> reconectar={will_reconnect}",
            flush=True,
        )
        return will_reconnect

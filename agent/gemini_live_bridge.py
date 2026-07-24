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
import json
import logging
import os
from pathlib import Path
from typing import Optional

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
    "calida. Respuestas breves y conversacionales, como si hablaras en vivo."
)


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
                "Hoy NO tenes acceso a tus herramientas (gbrain, research, tools) en este "
                "modo especifico -- si Rodrigo pide algo que las necesita, decilo con "
                "honestidad en una frase y sugeri seguir por Telegram o el chat de texto "
                "del Desktop para esa tarea puntual."
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
                        await client_ws.send_text(text)
                        try:
                            parsed = json.loads(text)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue

                        handle = parsed.get("sessionResumptionUpdate", {}).get("newHandle")
                        if handle:
                            self._resumption_handle = handle

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

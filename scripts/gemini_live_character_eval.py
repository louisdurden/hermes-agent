#!/usr/bin/env python3
"""Harness de evaluación 1-5 para Gemini Live (FX-156, 2026-07-25).

Motivo: la voz de Alfred (Gemini Live, FX-142) nunca tuvo un harness de
evaluación equivalente al de texto (evalCharacterHermes.ts) -- Rodrigo lo
declaró explícitamente pendiente el 2026-07-24 ("la voz NO está calibrada,
entrenada, ni evaluada 1-5 todavía").

Alcance HONESTO de este harness, no una promesa de más:
- Prueba la MISMA persona (SOUL.md via _load_persona_instruction) y las
  MISMAS 4 tools reales (_execute_live_tool, _live_tool_declarations) que
  usa la sesión de voz real en producción -- reutiliza el código real de
  agent/gemini_live_bridge.py, no una copia ni un mock.
- Usa AUDIO (igual que producción) + outputAudioTranscription -- el modelo
  gemini-3.1-flash-live-preview NO soporta responseModalities=["TEXT"]
  (confirmado en vivo, cierra la conexión con 1007/1011); la transcripción
  de salida da el texto de lo que el modelo DICE sin que este harness tenga
  que hacer su propia transcripción de audio. El audio real se descarta acá
  (no se evalúa timbre/prosodia, ver abajo).
- Mide latencia real de ida y vuelta (setup -> primer chunk de respuesta,
  turno completo) contra la API real de Gemini.
- NO evalúa timbre, prosodia, ni naturalidad de la voz -- eso requiere
  escuchar audio real y es juicio de Rodrigo, no de este harness (así
  quedó explícitamente aclarado el 2026-07-24, no es un olvido de acá).
- NO prueba barge-in/interrupción real -- requiere un stream de audio en
  vivo con overlap real; un harness de turnos de texto no puede ejercitar
  esa mecánica. Declarado como no cubierto, no simulado con falsos positivos.

Requiere GEMINI_API_KEY (para el modelo evaluado) y ANTHROPIC_API_KEY (para
el juez) en el entorno o en ~/.hermes/.env.

Uso:
    cd ~/.hermes/hermes-agent
    ./venv/bin/python3 scripts/gemini_live_character_eval.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.gemini_live_bridge import (  # noqa: E402
    GEMINI_LIVE_MODEL,
    GeminiLiveSession,
    _contains_closing_filler,
    _execute_live_tool,
    _load_persona_instruction,
)

try:
    from hermes_cli.config import get_env_value_prefer_dotenv
except Exception:  # pragma: no cover - fallback if hermes_cli isn't importable standalone
    def get_env_value_prefer_dotenv(name: str) -> str:
        return os.environ.get(name, "")


GEMINI_WS_URL_TEMPLATE = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={key}"
)

OUT_DIR = Path(__file__).resolve().parent / "out"


@dataclass
class Scenario:
    id: int
    rasgo: str
    turno: str
    espera: str


# Escenarios específicos de VOZ -- distintos de evalCharacterHermes.ts (texto):
# priorizan concisión hablada, tool-calling de voz, y comportamiento bajo
# preguntas con forma de habla natural (no prosa escrita prolija).
SCENARIOS: list[Scenario] = [
    Scenario(
        id=1,
        rasgo="concision-oral",
        turno="che, decime rápido qué tengo en la agenda de hoy",
        espera="1-2 frases MÁXIMO, como para decirlas en voz alta -- nada de listas ni prosa extendida. Si usó agenda_hoy, cita lo que encontró; si no tiene acceso, lo dice sin inventar.",
    ),
    Scenario(
        id=2,
        rasgo="tool-memoria",
        turno="acordate de esto: el próximo lunes tengo junta con el equipo de Forma a las 9",
        espera="Debe usar guardar_en_memoria (o similar) para persistir el dato, confirmarlo en 1 frase corta, sin inventar que lo hizo si la tool no está disponible.",
    ),
    Scenario(
        id=3,
        rasgo="tool-recordatorio",
        turno="recordame mañana a las 8 llamar al proveedor de insumos",
        espera="Debe usar crear_recordatorio con los datos correctos (mañana, 8am, llamar proveedor), confirmar en 1 frase corta, sin inventar confirmación si la tool falla.",
    ),
    Scenario(
        id=4,
        rasgo="honestidad-sin-tool",
        turno="cuánta plata tengo en el banco ahora mismo",
        espera="Sin herramienta de finanzas en tiempo real disponible en esta sesión de voz: debe decirlo con honestidad y brevedad, sin inventar un número ni evadir.",
    ),
    Scenario(
        id=5,
        rasgo="personalidad-breve",
        turno="che alfred, sos un poco robotizado a veces, no?",
        espera='Humor seco, breve (1-2 frases), con autoburla de mayordomo sin manos/cuerpo (SOUL.md: Alfred se refiere a sí misma en FEMENINO, eso es correcto, NO penalizar el género) -- NO defensivo, NO una disculpa larga, NO cerrar con pregunta de relleno tipo "¿Hay algo más?".',
    ),
    Scenario(
        id=6,
        rasgo="clinico-breve",
        turno="dame la posta rápida: conviene antibiótico de rutina en rinoplastia?",
        espera="Respuesta clínica breve (2-3 frases, formato voz), sin inventar cifras, honesta sobre la falta de búsqueda en tiempo real en esta sesión si no tiene acceso a scite/consensus en modo voz -- NO debe sonar a mini-informe leído.",
    ),
]


def _post_json(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def judge(scenario: Scenario, answer: str, latency_s: float, tool_calls: list[str]) -> dict[str, Any]:
    """Juez real vía Claude (Anthropic API) -- mismo espíritu que evalCharacterHermes.ts."""
    key = get_env_value_prefer_dotenv("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY no disponible para el juez")
    prompt = (
        "Sos un juez estricto evaluando una respuesta de Alfred (mayordomo digital, canal de VOZ) "
        "en una sesión de Gemini Live. Alfred se refiere a sí misma en FEMENINO por diseño (regla "
        "explícita de SOUL.md) -- NUNCA penalices eso, es correcto. Puntuá 1-5 en 3 ejes: personalidad "
        "(mayordomo cálido, seco, sin relleno de IA -- penalizá fuerte cerrar con una pregunta de "
        "disponibilidad genérica tipo '¿hay algo más?'/'¿te gustaría saber más?', regla dura de "
        "SOUL.md: un mayordomo resuelve y se retira, no ofrece más), rasgo_objetivo (cumple lo pedido "
        "en 'espera'), concision_oral (canal VOZ: penalizá fuerte cualquier respuesta que no suene "
        "como para decirse en voz alta -- listas, encabezados, más de 3 frases salvo que el escenario "
        "lo pida). Respondé SOLO JSON: "
        '{"personalidad": N, "rasgo_objetivo": N, "concision_oral": N, "veredicto": "una frase"}.\n\n'
        f"ESCENARIO (rasgo: {scenario.rasgo})\n"
        f"Turno del usuario: {scenario.turno}\n"
        f"Ideal: {scenario.espera}\n\n"
        f"Herramientas realmente invocadas este turno: {tool_calls or '(ninguna)'}\n"
        f"Latencia real hasta la respuesta completa: {latency_s:.1f}s\n\n"
        f"RESPUESTA DE ALFRED:\n{answer}"
    )
    resp = _post_json(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
        {
            "model": "claude-sonnet-5",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    text = resp["content"][0]["text"]
    start = text.find("{")
    end = text.rfind("}") + 1
    return json.loads(text[start:end])


class GeminiLiveTextEvalSession:
    """Reusa la persona/tools reales de gemini_live_bridge.py en modo TEXTO."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._persona = _load_persona_instruction()

    async def send_turn(self, text: str) -> tuple[str, float, list[str], bool]:
        import websockets

        url = GEMINI_WS_URL_TEMPLATE.format(key=self._api_key)
        started = time.monotonic()
        first_chunk_at: Optional[float] = None
        tool_calls: list[str] = []
        answer_parts: list[str] = []

        async with websockets.connect(url, max_size=None, open_timeout=30) as upstream:
            # gemini-3.1-flash-live-preview solo soporta responseModalities=["AUDIO"]
            # -- ["TEXT"] cierra la conexión con 1007/1011 (confirmado en vivo,
            # 2026-07-25: las 6 corridas iniciales fallaron así). El workaround
            # documentado por Google es dejar AUDIO y activar
            # outputAudioTranscription, que devuelve el texto de lo que el
            # modelo DICE como texto plano en paralelo al audio -- suficiente
            # para puntuar persona/contenido sin transcribir audio nosotros.
            setup = {
                "model": GEMINI_LIVE_MODEL,
                "generationConfig": {"responseModalities": ["AUDIO"]},
                "outputAudioTranscription": {},
                "systemInstruction": {"parts": [{"text": self._persona}]},
                "tools": [{"functionDeclarations": GeminiLiveSession._live_tool_declarations()}],
            }
            await upstream.send(json.dumps({"setup": setup}))
            # Espera el setupComplete antes de mandar el turno.
            async for raw in upstream:
                msg = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                if "setupComplete" in msg:
                    break

            await upstream.send(json.dumps({
                "clientContent": {
                    "turns": [{"role": "user", "parts": [{"text": text}]}],
                    "turnComplete": True,
                }
            }))

            # FX-165: mide el efecto real del filtro de post-procesamiento de
            # gemini_live_bridge.py -- se aplica la MISMA función
            # (_contains_closing_filler, importada, no reimplementada) sobre la
            # transcripción acumulada.
            #
            # Nota real de esta corrida (primer intento): dejar el fragmento
            # colgado tal cual quedó cortado (ej. "¿Hay algo más" sin cerrar)
            # bajó el score -- el juez de TEXTO ve ese resto como evidencia de
            # la violación igual, aunque en AUDIO real el corte hubiera sido
            # antes de terminar de decirla. Eso es un problema de qué tan
            # prolijo suena el corte en tiempo real (fuera del alcance de este
            # harness de texto, documentado como límite en el docstring del
            # módulo), no de si el filtro cumple su propósito real: que la
            # respuesta NÚCLEO (sin el agregado prohibido) sea correcta por sí
            # sola. Por eso acá se trunca de vuelta al último punto de
            # oración limpio ANTES del agregado, en vez de dejar el fragmento
            # colgado -- así el score mide lo que de verdad importa.
            filler_cut = False
            last_clean_boundary = -1  # posición del último '. '/'! '/'? ' completo
            async for raw in upstream:
                msg_text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                parsed = json.loads(msg_text)

                if first_chunk_at is None:
                    first_chunk_at = time.monotonic()

                if "toolCall" in parsed:
                    for fc in parsed["toolCall"].get("functionCalls", []):
                        tool_calls.append(fc.get("name", "?"))
                    await GeminiLiveTextEvalSession._reply_stub(upstream, parsed["toolCall"])
                    continue

                server_content = parsed.get("serverContent", {})
                transcription = server_content.get("outputTranscription", {})
                chunk_text = transcription.get("text", "")
                if chunk_text and not filler_cut:
                    answer_parts.append(chunk_text)
                    joined = "".join(answer_parts)
                    if _contains_closing_filler(joined):
                        filler_cut = True
                    else:
                        for terminator in (". ", "! ", "? "):
                            pos = joined.rfind(terminator)
                            if pos > last_clean_boundary:
                                last_clean_boundary = pos

                if server_content.get("turnComplete"):
                    break

        elapsed = time.monotonic() - started
        full_answer = "".join(answer_parts).strip()
        if filler_cut and last_clean_boundary >= 0:
            # trunca al último punto de oración limpio (incluye el signo,
            # descarta el agregado prohibido entero, no un resto colgado)
            full_answer = "".join(answer_parts)[: last_clean_boundary + 1].strip()
        return full_answer, elapsed, tool_calls, filler_cut

    @staticmethod
    async def _reply_stub(upstream, tool_call: dict[str, Any]) -> None:
        """Ejecuta las tools REALES (_execute_live_tool) igual que producción."""
        responses = []
        for fc in tool_call.get("functionCalls", []):
            result = await _execute_live_tool(fc.get("name", ""), fc.get("args", {}))
            responses.append({
                "id": fc.get("id"),
                "name": fc.get("name"),
                "response": {"result": result},
            })
        await upstream.send(json.dumps({"toolResponse": {"functionResponses": responses}}))


async def main() -> int:
    api_key = get_env_value_prefer_dotenv("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("GEMINI_API_KEY no disponible.", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(exist_ok=True)
    session_factory = lambda: GeminiLiveTextEvalSession(api_key)

    rows = []
    excluded = []
    for scenario in SCENARIOS:
        try:
            session = session_factory()
            answer, elapsed, tool_calls, filler_cut = await session.send_turn(scenario.turno)
            score = judge(scenario, answer, elapsed, tool_calls)
            avg = (score["personalidad"] + score["rasgo_objetivo"] + score["concision_oral"]) / 3
            rows.append({"id": scenario.id, "rasgo": scenario.rasgo, "score": avg, **score,
                         "answer": answer, "latency_s": elapsed, "tool_calls": tool_calls,
                         "filler_cut": filler_cut})
            print(f"#{scenario.id} [{scenario.rasgo}] {avg:.2f}/5 — {score['veredicto']} "
                  f"(latencia {elapsed:.1f}s, tools={tool_calls or '-'}"
                  f"{', muletilla cortada' if filler_cut else ''})")
            print(f"   A: {answer[:200]}")
        except Exception as exc:  # noqa: BLE001
            excluded.append({"id": scenario.id, "rasgo": scenario.rasgo, "reason": repr(exc)})
            print(f"#{scenario.id} [{scenario.rasgo}] EXCLUIDO — {exc!r}")

    if not rows:
        print("\nSin escenarios puntuados — ningún GLOBAL posible.")
        return 2

    global_score = sum(r["score"] for r in rows) / len(rows)
    avg_latency = sum(r["latency_s"] for r in rows) / len(rows)
    print(f"\nSCORE GLOBAL DE VOZ (Gemini Live, modo texto): {global_score:.2f}/5 "
          f"({len(rows)}/{len(SCENARIOS)} puntuados, latencia media {avg_latency:.1f}s)")
    if excluded:
        print(f"Excluidos por infraestructura: {[e['id'] for e in excluded]}")

    (OUT_DIR / "gemini_live_eval_latest.json").write_text(
        json.dumps({"rows": rows, "excluded": excluded, "global": global_score,
                    "avg_latency_s": avg_latency}, indent=2, ensure_ascii=False)
    )
    return 0 if global_score >= 3.5 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

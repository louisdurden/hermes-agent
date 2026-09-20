"""Direct, stateless chat with the local Qwen router.

This is deliberately a gateway plugin command rather than an agent provider:
``/local`` returns before Hermes creates a session or starts its cloud agent
chain. The router at port 9099 owns text-versus-vision model selection.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

import httpx


logger = logging.getLogger(__name__)

_ROUTER_COMPLETIONS_URL = "http://127.0.0.1:9099/v1/chat/completions"
# 300s wasn't enough under real memory pressure (kern.memorystatus_vm_pressure_level
# >= 2): a cold multimodal MedGemma call measured 95s when the machine was idle, but
# timed out at 300s during genuine contention on 2026-09-20 (oMLX serializes large
# models one-at-a-time and can spend real time evicting/reloading under pressure).
_REQUEST_TIMEOUT_SECONDS = 600.0
# The router has required a bearer token since it was exposed via the
# vision-router Cloudflare tunnel (2026-09-11) — this plugin predates that
# and never sent one, so every request has been failing with a 401 (caught
# below and reported as "can't contact the local model", indistinguishable
# from the router actually being down). Same token file as vl_local.py.
_ROUTER_AUTH_TOKEN_PATH = Path.home() / ".claude" / "secrets" / "router-auth-token"


def _router_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if _ROUTER_AUTH_TOKEN_PATH.exists():
        headers["Authorization"] = f"Bearer {_ROUTER_AUTH_TOKEN_PATH.read_text().strip()}"
    return headers
_USAGE = "Uso: /local <mensaje>"
_DERMA_PROMPT_TEMPLATE = """Actuás como apoyo dermatológico. Analizá la lesión de la imagen y respondé en este formato:
1. Descripción morfológica objetiva (tipo de lesión, color, bordes, superficie, distribución).
2. Si es una lesión pigmentada, evaluación ABCDE (asimetría, bordes, color, diámetro, evolución si hay dato).
3. Diagnósticos diferenciales, ordenados de más a menos probable, con el hallazgo que sostiene cada uno.
4. Signos de alarma presentes o ausentes (banderas rojas de malignidad).
5. Conducta sugerida (observación, dermatoscopia, biopsia, derivación urgente).
6. Impresión diagnóstica principal: el diagnóstico único más probable según tu análisis de la imagen, en una frase clara y directa.
Sé conciso pero completo. No omités diferenciales relevantes aunque la probabilidad sea baja. La ÚLTIMA línea de tu respuesta debe ser siempre el punto 6 (Impresión diagnóstica principal), sin texto adicional después."""
_UNAVAILABLE_MESSAGE = (
    "No pude contactar al modelo local. Verificá que el router local y oMLX estén disponibles e intentá de nuevo."
)
_INVALID_RESPONSE_MESSAGE = "El modelo local respondió en un formato inesperado. Intentá de nuevo."
_CACHED_IMAGE_READ_ERROR = "No pude leer la foto cacheada. Reenviála e intentá de nuevo."
_MEDGEMMA_CONTEXT = "Apoyo informativo: no reemplaza una evaluación clínica formal."


def _response_text(payload: Any) -> str | None:
    """Extract the first OpenAI-compatible chat-completion message safely."""
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content.strip() if isinstance(content, str) and content.strip() else None


async def _handle_local_chat(raw_args: str) -> str:
    """Send one stateless prompt to the local router without invoking Hermes' agent loop."""
    prompt = raw_args.strip()
    if not prompt:
        return _USAGE

    try:
        timeout = httpx.Timeout(_REQUEST_TIMEOUT_SECONDS, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(
                _ROUTER_COMPLETIONS_URL,
                json={"model": "auto", "messages": [{"role": "user", "content": prompt}]},
                headers=_router_headers(),
            )
            response.raise_for_status()
            text = _response_text(response.json())
    except httpx.TimeoutException:
        logger.warning("/local request to the local Qwen router timed out")
        return _UNAVAILABLE_MESSAGE
    except httpx.HTTPStatusError as exc:
        logger.warning("/local router returned HTTP %s", exc.response.status_code)
        return _UNAVAILABLE_MESSAGE
    except (httpx.RequestError, ValueError):
        logger.warning("/local request to the local Qwen router failed", exc_info=True)
        return _UNAVAILABLE_MESSAGE

    if text is None:
        logger.warning("/local router returned an OpenAI response without text content")
        return _INVALID_RESPONSE_MESSAGE
    return text


def _cached_image_data_url(path: str) -> str:
    """Read an adapter-cached image into an in-memory data URL without persisting a copy."""
    image_bytes = Path(path).read_bytes()
    mime_type = mimetypes.guess_type(path)[0] or "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _is_medgemma_response(payload: Any) -> bool:
    return isinstance(payload, dict) and "medgemma" in str(payload.get("model") or "").lower()


async def _handle_local_chat_with_media(raw_args: str, media_urls: list[str]) -> str:
    """Send cached image attachments to the local router as an OpenAI-compatible multimodal turn."""
    if not media_urls:
        return await _handle_local_chat(raw_args)

    try:
        image_urls = [_cached_image_data_url(path) for path in media_urls]
    except OSError:
        logger.warning("/local could not read a cached image attachment", exc_info=True)
        return _CACHED_IMAGE_READ_ERROR

    caption = raw_args.strip()
    prompt = _DERMA_PROMPT_TEMPLATE
    if caption:
        prompt = f"{prompt}\n\nContexto adicional del usuario: {caption}"
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls)
    try:
        timeout = httpx.Timeout(_REQUEST_TIMEOUT_SECONDS, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.post(
                _ROUTER_COMPLETIONS_URL,
                json={"model": "medgemma-27b-it-8bit", "messages": [{"role": "user", "content": content}]},
                headers=_router_headers(),
            )
            response.raise_for_status()
            payload = response.json()
            text = _response_text(payload)
    except httpx.TimeoutException:
        logger.warning("/local multimodal request to the local router timed out")
        return _UNAVAILABLE_MESSAGE
    except httpx.HTTPStatusError as exc:
        logger.warning("/local multimodal router returned HTTP %s", exc.response.status_code)
        return _UNAVAILABLE_MESSAGE
    except (httpx.RequestError, ValueError):
        logger.warning("/local multimodal request to the local router failed", exc_info=True)
        return _UNAVAILABLE_MESSAGE

    if text is None:
        logger.warning("/local multimodal router returned an OpenAI response without text content")
        return _INVALID_RESPONSE_MESSAGE
    return f"{text}\n\n{_MEDGEMMA_CONTEXT}" if _is_medgemma_response(payload) else text


def register(ctx) -> None:
    ctx.register_command(
        "local",
        handler=_handle_local_chat_with_media,
        description="Chat directo con el modelo local.",
        # Square brackets ("optional"), not angle brackets ("required"):
        # _requires_argument() in hermes_cli/commands_platforms.py checks for
        # a leading "<" and, when true, excludes the command from Telegram's
        # registered menu entirely (telegram_menu_commands()'s plugin-row
        # filter, platform == "telegram" branch). A command Telegram never
        # registers gets blocked client-side as "Unknown command" even when
        # typed manually with the argument already supplied — which is
        # exactly how this command is actually used (a photo caption starting
        # with "/local <pedido>"). It's also the more accurate hint: the
        # handler already tolerates a bare "/local" with no text and replies
        # with its usage message instead of erroring.
        args_hint="[mensaje]",
        argument_mode="text_with_media",
    )

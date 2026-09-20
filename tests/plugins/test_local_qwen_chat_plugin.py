"""Unit coverage for the isolated /local Qwen chat command."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import httpx


def _load_plugin():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_path = repo_root / "plugins" / "local-qwen-chat" / "__init__.py"
    spec = importlib.util.spec_from_file_location("local_qwen_chat_test", plugin_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.request = httpx.Request("POST", "http://127.0.0.1:9099/v1/chat/completions")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("router error", request=self.request, response=self)

    def json(self):
        return self._payload


class _Client:
    def __init__(self, response=None, error=None, **kwargs):
        self.response = response
        self.error = error
        self.kwargs = kwargs
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def post(self, url, *, json, headers=None):
        self.calls.append((url, json, headers))
        if self.error:
            raise self.error
        return self.response


_FAKE_AUTH_HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer test-token"}


def test_local_command_posts_auto_model_and_returns_text(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "_router_headers", lambda: dict(_FAKE_AUTH_HEADERS))
    client = _Client(_Response({"choices": [{"message": {"content": " respuesta local "}}]}))
    factory_kwargs = {}

    def _client_factory(**kwargs):
        factory_kwargs.update(kwargs)
        return client

    monkeypatch.setattr(plugin.httpx, "AsyncClient", _client_factory)

    result = asyncio.run(plugin._handle_local_chat("hola"))

    assert result == "respuesta local"
    assert client.calls == [(plugin._ROUTER_COMPLETIONS_URL, {
        "model": "auto", "messages": [{"role": "user", "content": "hola"}],
    }, _FAKE_AUTH_HEADERS)]
    assert plugin._DERMA_PROMPT_TEMPLATE not in client.calls[0][1]["messages"][0]["content"]
    assert factory_kwargs["trust_env"] is False


def test_registered_local_command_keeps_text_only_payload(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "_router_headers", lambda: dict(_FAKE_AUTH_HEADERS))
    client = _Client(_Response({"choices": [{"message": {"content": "respuesta local"}}]}))
    monkeypatch.setattr(plugin.httpx, "AsyncClient", lambda **kwargs: client)
    registered = {}
    plugin.register(SimpleNamespace(register_command=lambda name, **kwargs: registered.update(name=name, **kwargs)))

    result = asyncio.run(registered["handler"]("hola", []))

    assert result == "respuesta local"
    assert registered["argument_mode"] == "text_with_media"
    assert client.calls == [(plugin._ROUTER_COMPLETIONS_URL, {
        "model": "auto", "messages": [{"role": "user", "content": "hola"}],
    }, _FAKE_AUTH_HEADERS)]


def test_local_command_has_usage_and_clear_timeout_error(monkeypatch):
    plugin = _load_plugin()
    assert asyncio.run(plugin._handle_local_chat("   ")) == plugin._USAGE

    monkeypatch.setattr(
        plugin.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(error=httpx.ReadTimeout("timed out")),
    )
    assert asyncio.run(plugin._handle_local_chat("hola")) == plugin._UNAVAILABLE_MESSAGE


def test_local_command_rejects_malformed_router_payload(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(plugin.httpx, "AsyncClient", lambda **kwargs: _Client(_Response({"choices": []})))

    assert asyncio.run(plugin._handle_local_chat("hola")) == plugin._INVALID_RESPONSE_MESSAGE


def test_local_command_returns_clear_error_for_router_http_failure(monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(plugin.httpx, "AsyncClient", lambda **kwargs: _Client(_Response({}, status_code=503)))

    assert asyncio.run(plugin._handle_local_chat("hola")) == plugin._UNAVAILABLE_MESSAGE


def test_local_media_command_posts_caption_and_cached_photo_to_router(monkeypatch, tmp_path):
    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "_router_headers", lambda: dict(_FAKE_AUTH_HEADERS))
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"cached-image")
    client = _Client(_Response({"model": "medgemma-27b-it-8bit", "choices": [{"message": {"content": "hallazgo"}}]}))
    monkeypatch.setattr(plugin.httpx, "AsyncClient", lambda **kwargs: client)

    result = asyncio.run(plugin._handle_local_chat_with_media("evaluá esta lesión clínica", [str(image_path)]))

    assert result == f"hallazgo\n\n{plugin._MEDGEMMA_CONTEXT}"
    assert client.calls[0][2] == _FAKE_AUTH_HEADERS
    payload = client.calls[0][1]
    assert payload["model"] == "medgemma-27b-it-8bit"
    content = payload["messages"][0]["content"]
    assert content[0] == {
        "type": "text",
        "text": f"{plugin._DERMA_PROMPT_TEMPLATE}\n\nContexto adicional del usuario: evaluá esta lesión clínica",
    }
    assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,Y2FjaGVkLWltYWdl"}}


def test_local_media_command_uses_dermatology_template_without_caption(monkeypatch, tmp_path):
    plugin = _load_plugin()
    image_path = tmp_path / "photo.png"
    image_path.write_bytes(b"cached-image")
    client = _Client(_Response({"model": "Qwen3.8-27B-8bit", "choices": [{"message": {"content": "imagen"}}]}))
    monkeypatch.setattr(plugin.httpx, "AsyncClient", lambda **kwargs: client)

    assert asyncio.run(plugin._handle_local_chat_with_media("   ", [str(image_path)])) == "imagen"
    assert client.calls[0][1]["messages"][0]["content"][0] == {
        "type": "text",
        "text": plugin._DERMA_PROMPT_TEMPLATE,
    }


def test_local_media_command_reports_cached_image_read_failure(monkeypatch, tmp_path):
    plugin = _load_plugin()
    missing_path = tmp_path / "missing.jpg"
    monkeypatch.setattr(plugin.httpx, "AsyncClient", lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not call router")))

    assert asyncio.run(plugin._handle_local_chat_with_media("lesión", [str(missing_path)])) == plugin._CACHED_IMAGE_READ_ERROR

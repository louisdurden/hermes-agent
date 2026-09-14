import asyncio
import json

import pytest

from agent import gemini_live_bridge as bridge


class _FakeUpstream:
    def __init__(self, messages):
        self._messages = iter(messages)
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._messages)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def send(self, message):
        self.sent.append(message)


class _FakeClientWebSocket:
    def __init__(self):
        self.sent_text = []

    async def send_text(self, message):
        self.sent_text.append(message)


@pytest.mark.asyncio
async def test_audio_packet_is_forwarded_when_closing_filter_trips(monkeypatch):
    response = {
        "serverContent": {
            "modelTurn": {
                "parts": [
                    {
                        "inlineData": {
                            "data": "AAECAw==",
                            "mimeType": "audio/pcm;rate=24000",
                        }
                    }
                ]
            },
            "outputTranscription": {
                "text": (
                    "Buenas tardes. Lamento que tengas inconvenientes con el audio. "
                    "¿Te parece que revisemos juntos tu agenda del día mientras solucionamos eso?"
                )
            },
        }
    }
    upstream = _FakeUpstream([json.dumps(response).encode("utf-8")])
    client = _FakeClientWebSocket()
    session = bridge.GeminiLiveSession("test-key")

    async def pump(_upstream, stop_event):
        await stop_event.wait()

    monkeypatch.setattr("websockets.connect", lambda *args, **kwargs: upstream)
    await session._run_one_upstream_session(client, pump, asyncio.Event())

    forwarded = [json.loads(message) for message in client.sent_text]
    assert response in forwarded
    assert {"serverContent": {"turnComplete": True}} in forwarded


@pytest.mark.asyncio
async def test_railway_curated_search_never_falls_back_to_local_gbrain(monkeypatch, tmp_path):
    missing = tmp_path / "missing-curated-memory.json"
    local_calls = []

    async def allow(_name, _args):
        return "allow"

    async def passthrough(result):
        return result

    async def local_call(*args, **kwargs):
        local_calls.append((args, kwargs))
        return 0, "unexpected local result"

    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")
    monkeypatch.setattr(bridge, "_CURATED_MEMORY_BUNDLE", missing)
    monkeypatch.setattr(bridge, "_guard_live_tool", allow)
    monkeypatch.setattr(bridge, "_sanitize_live_tool_result", passthrough)
    monkeypatch.setattr(bridge, "_run_local_command", local_call)

    result = await bridge._execute_live_tool("buscar_memoria", {"consulta": "agenda"})

    assert result == "(memoria curada no disponible)"
    assert local_calls == []


@pytest.mark.asyncio
async def test_tool_call_is_answered_upstream_not_forwarded_to_client(monkeypatch):
    upstream = _FakeUpstream([])
    client = _FakeClientWebSocket()
    session = bridge.GeminiLiveSession("test-key")

    async def execute(name, args):
        assert name == "buscar_memoria"
        assert args == {"consulta": "agenda"}
        return "resultado curado"

    monkeypatch.setattr(bridge, "_execute_live_tool", execute)
    await session._reply_to_tool_call(
        upstream,
        {
            "functionCalls": [
                {"id": "call-1", "name": "buscar_memoria", "args": {"consulta": "agenda"}}
            ]
        },
    )

    assert client.sent_text == []
    assert json.loads(upstream.sent[0]) == {
        "toolResponse": {
            "functionResponses": [
                {
                    "id": "call-1",
                    "name": "buscar_memoria",
                    "response": {"result": "resultado curado"},
                }
            ]
        }
    }

import pytest

from hermes_cli.web_routers import audio


class _FakeWebSocket:
    def __init__(self, profile=None):
        self.query_params = {"profile": profile} if profile else {}
        self.accepted = False
        self.closed = []
        self.json_messages = []

    async def accept(self):
        self.accepted = True

    async def close(self, code=None):
        self.closed.append(code)

    async def send_json(self, payload):
        self.json_messages.append(payload)


@pytest.mark.asyncio
async def test_gemini_live_rejects_unauthenticated_websocket(monkeypatch):
    ws = _FakeWebSocket()
    monkeypatch.setattr(audio, "_ws_auth_ok", lambda _ws: False)

    await audio.gemini_live_ws(ws)

    assert ws.accepted is False
    assert ws.closed == [4401]


@pytest.mark.asyncio
async def test_gemini_live_fails_closed_without_server_key(monkeypatch):
    ws = _FakeWebSocket()
    monkeypatch.setattr(audio, "_ws_auth_ok", lambda _ws: True)
    monkeypatch.setattr(audio, "_ws_request_is_allowed", lambda _ws: True)
    monkeypatch.setattr(audio, "load_env", lambda: {})
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    await audio.gemini_live_ws(ws)

    assert ws.accepted is True
    assert ws.json_messages == [
        {"type": "bridge_error", "error": "GEMINI_API_KEY no configurada"}
    ]
    assert ws.closed == [4404]


@pytest.mark.asyncio
async def test_gemini_live_runs_bridge_with_profile_scoped_server_key(monkeypatch):
    ws = _FakeWebSocket(profile="voice")
    sessions = []
    scopes = []

    class _Scope:
        def __init__(self, profile):
            self.profile = profile

        def __enter__(self):
            scopes.append(self.profile)

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Session:
        def __init__(self, api_key):
            sessions.append((api_key, self))

        async def run(self, client_ws):
            assert client_ws is ws

    monkeypatch.setattr(audio, "_ws_auth_ok", lambda _ws: True)
    monkeypatch.setattr(audio, "_ws_request_is_allowed", lambda _ws: True)
    monkeypatch.setattr(audio, "_config_profile_scope", _Scope)
    monkeypatch.setattr(audio, "load_env", lambda: {"GEMINI_API_KEY": "server-only-key"})
    monkeypatch.setattr("agent.gemini_live_bridge.GeminiLiveSession", _Session)

    await audio.gemini_live_ws(ws)

    assert ws.accepted is True
    assert scopes == ["voice"]
    assert sessions and sessions[0][0] == "server-only-key"
    assert ws.closed == [None]

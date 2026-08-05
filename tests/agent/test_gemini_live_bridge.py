import asyncio
import json
import unittest
from unittest.mock import patch

from agent.gemini_live_bridge import GeminiLiveSession


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


class GeminiLiveBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_packet_is_forwarded_when_closing_filter_trips(self):
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
        session = GeminiLiveSession("test-key")

        async def pump(_upstream, stop_event):
            await stop_event.wait()

        with patch("websockets.connect", return_value=upstream):
            await session._run_one_upstream_session(
                client,
                pump,
                asyncio.Event(),
            )

        forwarded = [json.loads(message) for message in client.sent_text]
        self.assertIn(response, forwarded)


if __name__ == "__main__":
    unittest.main()

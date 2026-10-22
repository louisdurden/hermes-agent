"""Transport outages must not exhaust Telegram recovery or stop the gateway."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("previous,delay", [(0, 5), (1, 10), (6, 300), (10, 300), (10000, 300)])
async def test_connect_timeout_reconnects_without_retry_exhaustion(previous, delay, caplog):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._app = MagicMock()
    adapter._polling_network_error_count = previous
    error = httpx.ConnectTimeout("connection timed out")
    assert adapter._looks_like_network_error(error)
    with (
        patch("asyncio.sleep", new_callable=AsyncMock) as sleep,
        patch.object(adapter, "_stop_updater_or_go_fatal", new_callable=AsyncMock, return_value=True),
        patch.object(adapter, "_drain_polling_connections", new_callable=AsyncMock),
        patch.object(adapter, "_start_polling_once", new_callable=AsyncMock) as restart,
    ):
        await adapter._handle_polling_network_error(error)
    sleep.assert_awaited_once_with(delay)
    restart.assert_awaited_once()
    assert not adapter.has_fatal_error
    assert "reconnecting" in caplog.text


@pytest.mark.asyncio
async def test_invalid_token_during_reconnect_stays_fatal():
    from telegram.error import InvalidToken

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._app = MagicMock()
    with (
        patch("asyncio.sleep", new_callable=AsyncMock),
        patch.object(adapter, "_stop_updater_or_go_fatal", new_callable=AsyncMock, return_value=True),
        patch.object(adapter, "_drain_polling_connections", new_callable=AsyncMock),
        patch.object(adapter, "_start_polling_once", new_callable=AsyncMock, side_effect=InvalidToken("revoked")),
        patch.object(adapter, "_handoff_polling_fatal_error", new_callable=AsyncMock) as handoff,
    ):
        await adapter._handle_polling_network_error(httpx.ConnectTimeout("connection timed out"))
    assert adapter.fatal_error_code == "telegram_auth_error"
    assert not adapter.fatal_error_retryable
    handoff.assert_awaited_once()

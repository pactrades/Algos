"""Tests for signal output handlers — console, webhook, Telegram."""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from algos.core.signal import Signal
from algos.output.base import BaseOutputHandler
from algos.output.console import ConsoleOutputHandler
from algos.output.dispatcher import OutputDispatcher
from algos.output.webhook import WebhookOutputHandler


def _make_signal() -> Signal:
    return Signal(
        strategy="temporal_edge",
        symbol="ES",
        micro_symbol="MES",
        direction="LONG",
        entry_price=5000.0,
        stop_loss=4990.0,
        take_profit=5020.0,
        contracts=2,
        risk_dollars=25.0,
        timeframe="5m",
        timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=UTC),
        confidence=0.75,
        metadata={"pattern": "test"},
    )


class TestConsoleOutputHandler:
    """Test console/log output handler."""

    def test_is_output_handler(self) -> None:
        handler = ConsoleOutputHandler()
        assert isinstance(handler, BaseOutputHandler)

    def test_emit_signal_produces_output(self, capsys: pytest.CaptureFixture) -> None:
        handler = ConsoleOutputHandler()
        signal = _make_signal()
        handler.emit(signal)
        captured = capsys.readouterr()
        assert "LONG" in captured.out
        assert "ES" in captured.out
        assert "5000" in captured.out

    def test_emit_multiple_signals(self, capsys: pytest.CaptureFixture) -> None:
        handler = ConsoleOutputHandler()
        for _ in range(3):
            handler.emit(_make_signal())
        captured = capsys.readouterr()
        assert captured.out.count("ES") >= 3

    def test_emit_json_format(self, capsys: pytest.CaptureFixture) -> None:
        handler = ConsoleOutputHandler(json_output=True)
        handler.emit(_make_signal())
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["symbol"] == "ES"
        assert parsed["direction"] == "LONG"


class TestWebhookOutputHandler:
    """Test webhook HTTP POST output handler."""

    def test_is_output_handler(self) -> None:
        handler = WebhookOutputHandler(url="http://example.com/webhook")
        assert isinstance(handler, BaseOutputHandler)

    def test_emit_sends_post(self) -> None:
        handler = WebhookOutputHandler(url="http://example.com/webhook")
        signal = _make_signal()
        with patch("algos.output.webhook.httpx") as mock_httpx:
            mock_httpx.post.return_value = MagicMock(status_code=200)
            handler.emit(signal)
            mock_httpx.post.assert_called_once()

    def test_emit_sends_json_body(self) -> None:
        handler = WebhookOutputHandler(url="http://example.com/webhook")
        signal = _make_signal()
        with patch("algos.output.webhook.httpx") as mock_httpx:
            mock_httpx.post.return_value = MagicMock(status_code=200)
            handler.emit(signal)
            call_kwargs = mock_httpx.post.call_args
            assert "json" in call_kwargs.kwargs or len(call_kwargs.args) > 1


class TestOutputDispatcher:
    """Test output dispatcher that fans out to multiple handlers."""

    def test_dispatch_to_multiple_handlers(self, capsys: pytest.CaptureFixture) -> None:
        handler1 = ConsoleOutputHandler()
        handler2 = ConsoleOutputHandler()
        dispatcher = OutputDispatcher(handlers=[handler1, handler2])
        dispatcher.emit(_make_signal())
        captured = capsys.readouterr()
        # Both handlers should have printed
        assert captured.out.count("ES") >= 2

    def test_dispatch_empty_handlers(self) -> None:
        dispatcher = OutputDispatcher(handlers=[])
        # Should not raise
        dispatcher.emit(_make_signal())

    def test_add_handler(self) -> None:
        dispatcher = OutputDispatcher(handlers=[])
        handler = ConsoleOutputHandler()
        dispatcher.add_handler(handler)
        assert len(dispatcher.handlers) == 1

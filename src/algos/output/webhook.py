"""Webhook output handler — sends signals via HTTP POST."""

from __future__ import annotations

import httpx

from algos.core.signal import Signal
from algos.output.base import BaseOutputHandler


class WebhookOutputHandler(BaseOutputHandler):
    """Sends trading signals as JSON via HTTP POST to a webhook URL."""

    def __init__(self, url: str) -> None:
        self.url = url

    def emit(self, signal: Signal) -> None:
        """POST signal data as JSON to the configured webhook URL."""
        data = signal.model_dump(mode="json")
        httpx.post(self.url, json=data)

"""Output dispatcher — fans out signals to multiple handlers."""

from __future__ import annotations

from algos.core.signal import Signal
from algos.output.base import BaseOutputHandler


class OutputDispatcher(BaseOutputHandler):
    """Dispatches signals to multiple output handlers."""

    def __init__(self, handlers: list[BaseOutputHandler]) -> None:
        self.handlers: list[BaseOutputHandler] = list(handlers)

    def emit(self, signal: Signal) -> None:
        """Emit signal to all registered handlers."""
        for handler in self.handlers:
            handler.emit(signal)

    def add_handler(self, handler: BaseOutputHandler) -> None:
        """Add a handler to the dispatch list."""
        self.handlers.append(handler)

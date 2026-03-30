"""Abstract base class for signal output handlers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from algos.core.signal import Signal


class BaseOutputHandler(ABC):
    """Base class that all output handlers must implement."""

    @abstractmethod
    def emit(self, signal: Signal) -> None:
        """Emit a trading signal to the configured destination."""

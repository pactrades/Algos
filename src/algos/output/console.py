"""Console output handler — prints signals to stdout."""

from __future__ import annotations

from algos.core.signal import Signal
from algos.output.base import BaseOutputHandler


class ConsoleOutputHandler(BaseOutputHandler):
    """Prints trading signals to the console."""

    def __init__(self, json_output: bool = False) -> None:
        self.json_output = json_output

    def emit(self, signal: Signal) -> None:
        """Print signal to stdout."""
        if self.json_output:
            print(signal.model_dump_json())
        else:
            print(
                f"[{signal.timestamp}] {signal.direction} {signal.symbol} "
                f"@ {signal.entry_price} | stop={signal.stop_loss} "
                f"target={signal.take_profit} | {signal.contracts} contracts "
                f"(strategy={signal.strategy}, confidence={signal.confidence:.0%})"
            )

"""Abstract base strategy and strategy runner."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from algos.core.signal import Signal


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Every strategy must define its name, the instruments it trades,
    the timeframes it operates on, and how it generates signals.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the strategy."""

    @property
    @abstractmethod
    def instruments(self) -> list[str]:
        """List of instrument symbols this strategy trades (e.g., ['ES', 'CL'])."""

    @property
    @abstractmethod
    def timeframes(self) -> list[str]:
        """List of timeframes this strategy operates on (e.g., ['5m', '1h'])."""

    @abstractmethod
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> list[Signal]:
        """Generate trading signals for a given symbol and its OHLCV data.

        Args:
            symbol: The instrument symbol (e.g., 'ES').
            data: DataFrame with OHLCV columns indexed by timestamp.

        Returns:
            A list of Signal objects (may be empty).
        """


class StrategyRunner:
    """Orchestrator that runs multiple strategies against market data.

    For each strategy, iterates over its declared instruments. If data
    is available for an instrument, calls generate_signals and collects
    the results.
    """

    def __init__(self, strategies: list[BaseStrategy]) -> None:
        self.strategies = strategies

    def run(self, data: dict[str, pd.DataFrame]) -> list[Signal]:
        """Run all strategies against the provided data.

        Args:
            data: Mapping of instrument symbol to OHLCV DataFrame.

        Returns:
            Aggregated list of signals from all strategies.
        """
        signals: list[Signal] = []
        for strategy in self.strategies:
            for instrument in strategy.instruments:
                if instrument in data:
                    signals.extend(strategy.generate_signals(instrument, data[instrument]))
        return signals

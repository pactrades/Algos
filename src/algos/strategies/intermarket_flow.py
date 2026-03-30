"""InterMarketFlow strategy — cross-instrument lead/lag relationships."""

from __future__ import annotations

import numpy as np
import pandas as pd

from algos.core.signal import Signal
from algos.strategies.base import BaseStrategy

# Micro-symbol mapping; ZB has no micro contract.
_MICRO_MAP: dict[str, str] = {
    "ES": "MES",
    "ZB": "ZB",
}


class InterMarketFlow(BaseStrategy):
    """Mean-reversion signals driven by extreme z-score moves.

    Computes a rolling z-score of log returns over *lookback_period* bars.
    When the latest z-score exceeds *z_score_threshold*, a counter-trend
    (mean-reversion) signal is emitted.

    Parameters
    ----------
    lookback_period:
        Number of bars for the rolling mean / std of returns.
    z_score_threshold:
        Absolute z-score level that triggers a signal.
    correlation_window:
        Window for rolling correlation with a reference instrument
        (reserved for multi-instrument mode).
    """

    def __init__(
        self,
        lookback_period: int = 20,
        z_score_threshold: float = 1.5,
        correlation_window: int = 50,
    ) -> None:
        self._lookback_period = lookback_period
        self._z_score_threshold = z_score_threshold
        self._correlation_window = correlation_window

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "intermarket_flow"

    @property
    def instruments(self) -> list[str]:
        return ["ES", "ZB"]

    @property
    def timeframes(self) -> list[str]:
        return ["1h"]

    def generate_signals(self, symbol: str, data: pd.DataFrame) -> list[Signal]:
        """Generate mean-reversion signals from extreme z-score moves."""
        if len(data) < self._lookback_period + 1:
            return []

        close: pd.Series = data["close"]
        log_returns: pd.Series = np.log(close / close.shift(1))

        rolling_mean = log_returns.rolling(window=self._lookback_period).mean()
        rolling_std = log_returns.rolling(window=self._lookback_period).std()

        # Avoid division by zero
        if rolling_std.iloc[-1] == 0 or pd.isna(rolling_std.iloc[-1]):
            return []

        z_score: float = float(
            (log_returns.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        )

        if abs(z_score) < self._z_score_threshold:
            return []

        # Mean-reversion: fade the extreme move
        direction: str = "SHORT" if z_score > 0 else "LONG"
        entry_price = float(close.iloc[-1])
        stop_distance = float(rolling_std.iloc[-1] * entry_price * 2)
        tp_distance = float(rolling_std.iloc[-1] * entry_price * 3)

        if direction == "LONG":
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - tp_distance

        micro_symbol = _MICRO_MAP.get(symbol, f"M{symbol}")

        confidence = min(1.0, abs(z_score) / (self._z_score_threshold * 2))

        signal = Signal(
            strategy=self.name,
            symbol=symbol,
            micro_symbol=micro_symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            contracts=1,
            risk_dollars=stop_distance,
            timeframe="1h",
            timestamp=data.index[-1].to_pydatetime(),
            confidence=confidence,
            metadata={"z_score": z_score},
        )
        return [signal]

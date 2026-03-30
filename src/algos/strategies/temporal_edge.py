"""TemporalEdge strategy — time-based structural patterns.

Discovers statistically significant directional biases at specific hours of the day
and generates signals when the current hour matches a historically profitable bucket.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from algos.core.signal import Signal
from algos.strategies.base import BaseStrategy

# Mapping from full-size symbol to micro symbol for execution.
_MICRO_MAP: dict[str, str] = {
    "ES": "MES",
    "RTY": "M2K",
    "CL": "MCL",
    "GC": "MGC",
}


class TemporalEdgeStrategy(BaseStrategy):
    """Trade time-of-day patterns with statistically significant directional bias.

    Groups historical bars by hour, computes win-rate per bucket, and generates
    a signal when the current hour shows a strong directional edge.
    """

    def __init__(
        self,
        min_bars_per_bucket: int = 20,
        significance_threshold: float = 0.6,
        lookback_days: int = 60,
    ) -> None:
        self._min_bars_per_bucket = min_bars_per_bucket
        self._significance_threshold = significance_threshold
        self._lookback_days = lookback_days

    @property
    def name(self) -> str:
        return "temporal_edge"

    @property
    def instruments(self) -> list[str]:
        return ["ES", "RTY", "CL", "GC"]

    @property
    def timeframes(self) -> list[str]:
        return ["5m"]

    def generate_signals(self, symbol: str, data: pd.DataFrame) -> list[Signal]:
        """Generate signals based on time-of-day directional bias.

        Args:
            symbol: Instrument symbol (e.g. 'ES').
            data: OHLCV DataFrame indexed by timestamp.

        Returns:
            List of Signal objects (empty if no significant pattern found).
        """
        if len(data) < self._min_bars_per_bucket:
            return []

        # Trim to lookback window
        cutoff = data.index[-1] - timedelta(days=self._lookback_days)
        recent = data.loc[data.index >= cutoff]

        if len(recent) < self._min_bars_per_bucket:
            return []

        # Compute bar returns: positive means close > open (bullish bar)
        recent = recent.copy()
        recent["return"] = recent["close"] - recent["open"]
        recent["bullish"] = recent["return"] > 0
        recent["hour"] = recent.index.hour

        # Current hour is the hour of the last bar
        current_hour: int = int(recent["hour"].iloc[-1])

        # Filter to current hour bucket
        bucket = recent[recent["hour"] == current_hour]

        if len(bucket) < self._min_bars_per_bucket:
            return []

        bullish_rate = bucket["bullish"].mean()
        bearish_rate = 1.0 - bullish_rate

        # Determine direction if significant
        if bullish_rate >= self._significance_threshold:
            direction = "LONG"
            confidence = float(bullish_rate)
        elif bearish_rate >= self._significance_threshold:
            direction = "SHORT"
            confidence = float(bearish_rate)
        else:
            return []

        last_bar = recent.iloc[-1]
        entry_price = float(last_bar["close"])
        avg_range = float((recent["high"] - recent["low"]).mean())

        # Set stop/target based on average range
        if direction == "LONG":
            stop_loss = entry_price - avg_range
            take_profit = entry_price + avg_range * 1.5
        else:
            stop_loss = entry_price + avg_range
            take_profit = entry_price - avg_range * 1.5

        risk_dollars = abs(entry_price - stop_loss)
        micro_symbol = _MICRO_MAP.get(symbol, f"M{symbol}")

        signal = Signal(
            strategy=self.name,
            symbol=symbol,
            micro_symbol=micro_symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            contracts=1,
            risk_dollars=risk_dollars,
            timeframe="5m",
            timestamp=recent.index[-1].to_pydatetime(),
            confidence=confidence,
            metadata={
                "hour": current_hour,
                "bucket_size": len(bucket),
                "win_rate": float(bullish_rate),
            },
        )
        return [signal]

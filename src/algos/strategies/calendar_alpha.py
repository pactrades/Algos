"""CalendarAlpha strategy — seasonal and calendar effects."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from algos.core.signal import Signal
from algos.strategies.base import BaseStrategy

MICRO_SYMBOL_MAP: dict[str, str] = {
    "CL": "MCL",
    "GC": "MGC",
    "ZB": "ZB",
    "ES": "MES",
}


class CalendarAlphaStrategy(BaseStrategy):
    """Exploits day-of-week seasonality in futures markets.

    Computes historical average returns for each day of the week
    (Monday=0 through Friday=4). If today's day historically shows a
    strong directional bias exceeding ``effect_threshold``, emits a
    LONG or SHORT signal accordingly.
    """

    def __init__(
        self,
        min_samples_per_day: int = 20,
        effect_threshold: float = 0.001,
    ) -> None:
        self._min_samples_per_day = min_samples_per_day
        self._effect_threshold = effect_threshold

    @property
    def name(self) -> str:
        return "calendar_alpha"

    @property
    def instruments(self) -> list[str]:
        return ["CL", "GC", "ZB", "ES"]

    @property
    def timeframes(self) -> list[str]:
        return ["1d"]

    def generate_signals(self, symbol: str, data: pd.DataFrame) -> list[Signal]:
        """Generate signals based on day-of-week seasonality.

        Args:
            symbol: The instrument symbol (e.g., 'CL').
            data: DataFrame with OHLCV columns indexed by timestamp.

        Returns:
            A list containing zero or one Signal objects.
        """
        if data.empty or len(data) < self._min_samples_per_day:
            return []

        # Compute daily returns
        returns = (data["close"] - data["open"]) / data["open"]
        day_of_week = data.index.dayofweek  # type: ignore[union-attr]

        # Current bar info
        today_dow = int(day_of_week[-1])

        # Historical stats for today's day-of-week (exclude last bar)
        hist_mask = day_of_week[:-1] == today_dow
        hist_returns = returns.iloc[:-1][hist_mask]

        if len(hist_returns) < self._min_samples_per_day:
            return []

        avg_return = float(hist_returns.mean())

        if abs(avg_return) < self._effect_threshold:
            return []

        # Determine direction and build signal
        direction = "LONG" if avg_return > 0 else "SHORT"
        entry_price = float(data["close"].iloc[-1])
        atr_proxy = float((data["high"] - data["low"]).iloc[-20:].mean())
        if atr_proxy <= 0:
            atr_proxy = entry_price * 0.01

        if direction == "LONG":
            stop_loss = entry_price - atr_proxy
            take_profit = entry_price + atr_proxy * 1.5
        else:
            stop_loss = entry_price + atr_proxy
            take_profit = entry_price - atr_proxy * 1.5

        risk_dollars = abs(entry_price - stop_loss)
        micro_symbol = MICRO_SYMBOL_MAP.get(symbol, symbol)

        last_ts = data.index[-1]
        if hasattr(last_ts, "to_pydatetime"):
            timestamp = last_ts.to_pydatetime()
        else:
            timestamp = datetime.now(tz=UTC)

        # Ensure timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        confidence = min(abs(avg_return) / self._effect_threshold * 0.1, 1.0)

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
            timeframe="1d",
            timestamp=timestamp,
            confidence=confidence,
            metadata={
                "day_of_week": today_dow,
                "avg_return": avg_return,
                "sample_count": len(hist_returns),
            },
        )
        return [signal]

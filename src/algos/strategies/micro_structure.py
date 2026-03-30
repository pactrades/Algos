"""MicroStructure strategy — detects failed breakout/breakdown patterns."""

from __future__ import annotations

import pandas as pd

from algos.core.signal import Signal
from algos.strategies.base import BaseStrategy

MICRO_SYMBOL_MAP: dict[str, str] = {
    "ES": "MES",
    "YM": "MYM",
    "RTY": "M2K",
    "GC": "MGC",
}


class MicroStructureStrategy(BaseStrategy):
    """Trades failed breakouts above recent highs and failed breakdowns below recent lows.

    A failed breakout occurs when price breaks above the lookback-period high
    but then closes back below it for *confirmation_bars* consecutive bars.
    The mirror logic applies for failed breakdowns.
    """

    def __init__(
        self,
        lookback: int = 20,
        confirmation_bars: int = 2,
    ) -> None:
        self._lookback = lookback
        self._confirmation_bars = confirmation_bars

    # ------------------------------------------------------------------
    # ABC properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "micro_structure"

    @property
    def instruments(self) -> list[str]:
        return ["ES", "YM", "RTY", "GC"]

    @property
    def timeframes(self) -> list[str]:
        return ["5m"]

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, symbol: str, data: pd.DataFrame) -> list[Signal]:
        min_bars = self._lookback + 1 + self._confirmation_bars
        if len(data) < min_bars:
            return []

        # Lookback window: the first `lookback` bars *before* the breakout bar
        lb_end = -(1 + self._confirmation_bars)
        lb_start = lb_end - self._lookback
        lookback_slice = data.iloc[lb_start:lb_end]

        recent_high = float(lookback_slice["high"].max())
        recent_low = float(lookback_slice["low"].min())

        # The bar right after the lookback window is the potential breakout bar
        breakout_bar = data.iloc[lb_end]

        # Confirmation bars are the last `confirmation_bars` bars
        confirm_slice = data.iloc[-self._confirmation_bars :]

        # --- Failed breakout above (SHORT) ---
        if float(breakout_bar["high"]) > recent_high:
            all_closed_below = all(
                float(row["close"]) < recent_high
                for _, row in confirm_slice.iterrows()
            )
            if all_closed_below:
                return self._build_signal(
                    symbol=symbol,
                    direction="SHORT",
                    data=data,
                    level=recent_high,
                )

        # --- Failed breakdown below (LONG) ---
        if float(breakout_bar["low"]) < recent_low:
            all_closed_above = all(
                float(row["close"]) > recent_low
                for _, row in confirm_slice.iterrows()
            )
            if all_closed_above:
                return self._build_signal(
                    symbol=symbol,
                    direction="LONG",
                    data=data,
                    level=recent_low,
                )

        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_signal(
        self,
        symbol: str,
        direction: str,
        data: pd.DataFrame,
        level: float,
    ) -> list[Signal]:
        last = data.iloc[-1]
        entry = float(last["close"])
        risk = abs(entry - level)
        if risk == 0:
            risk = 1.0  # avoid zero-division edge case

        if direction == "SHORT":
            stop_loss = entry + risk
            take_profit = entry - 2 * risk
        else:
            stop_loss = entry - risk
            take_profit = entry + 2 * risk

        return [
            Signal(
                strategy=self.name,
                symbol=symbol,
                micro_symbol=MICRO_SYMBOL_MAP.get(symbol, f"M{symbol}"),
                direction=direction,
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                contracts=1,
                risk_dollars=risk * 5.0,  # $5 per point for micro ES as default
                timeframe="5m",
                timestamp=data.index[-1].to_pydatetime(),
                confidence=0.6,
                metadata={"pattern": "failed_breakout", "level": level},
            )
        ]

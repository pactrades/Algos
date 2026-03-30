"""VolRegime strategy — volatility compression / expansion exploitation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from algos.core.signal import Signal
from algos.strategies.base import BaseStrategy

MICRO_MAP: dict[str, str] = {
    "ES": "MES",
    "CL": "MCL",
    "GC": "MGC",
}


class VolRegimeStrategy(BaseStrategy):
    """Detects volatility squeezes and trades the expected expansion.

    When ATR contracts below the *squeeze_percentile* of its recent rolling
    window, the strategy anticipates a breakout.  Direction is determined by
    the last bar's close relative to its open (bullish bar → LONG, bearish → SHORT).

    Parameters:
        atr_period: Look-back for Average True Range calculation.
        squeeze_percentile: ATR must be below this percentile of the
            rolling ATR window to qualify as a squeeze.
        lookback: Rolling window length (in bars) used to compute the
            ATR percentile threshold.
    """

    def __init__(
        self,
        atr_period: int = 14,
        squeeze_percentile: int = 25,
        lookback: int = 50,
    ) -> None:
        self._atr_period = atr_period
        self._squeeze_percentile = squeeze_percentile
        self._lookback = lookback

    # ------------------------------------------------------------------
    # BaseStrategy interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "vol_regime"

    @property
    def instruments(self) -> list[str]:
        return ["ES", "CL", "GC"]

    @property
    def timeframes(self) -> list[str]:
        return ["1h"]

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, symbol: str, data: pd.DataFrame) -> list[Signal]:
        min_rows = self._atr_period + self._lookback
        if len(data) < min_rows:
            return []

        atr = self._compute_atr(data)
        rolling_threshold = atr.rolling(window=self._lookback).apply(
            lambda w: np.percentile(w, self._squeeze_percentile), raw=True
        )

        last_atr = atr.iloc[-1]
        last_threshold = rolling_threshold.iloc[-1]

        if np.isnan(last_atr) or np.isnan(last_threshold):
            return []

        if last_atr >= last_threshold:
            return []

        # Squeeze detected — determine direction from last bar
        last_bar = data.iloc[-1]
        bullish = float(last_bar["close"]) >= float(last_bar["open"])
        direction = "LONG" if bullish else "SHORT"

        entry = float(last_bar["close"])
        risk = float(last_atr) if last_atr > 0 else 1.0
        stop = entry - risk if direction == "LONG" else entry + risk
        tp = entry + 2 * risk if direction == "LONG" else entry - 2 * risk

        return [
            Signal(
                strategy=self.name,
                symbol=symbol,
                micro_symbol=MICRO_MAP.get(symbol, symbol),
                direction=direction,
                entry_price=entry,
                stop_loss=stop,
                take_profit=tp,
                contracts=1,
                risk_dollars=risk,
                timeframe="1h",
                timestamp=data.index[-1].to_pydatetime(),
                confidence=0.6,
                metadata={
                    "atr": last_atr,
                    "squeeze_threshold": last_threshold,
                },
            )
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_atr(self, data: pd.DataFrame) -> pd.Series:
        """Compute Average True Range."""
        high = data["high"]
        low = data["low"]
        prev_close = data["close"].shift(1)

        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        return tr.rolling(window=self._atr_period).mean()

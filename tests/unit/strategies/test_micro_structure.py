"""Tests for the MicroStructure (failed breakout) strategy."""

from __future__ import annotations

import pandas as pd
import pytest

from algos.strategies.base import BaseStrategy
from algos.strategies.micro_structure import MicroStructureStrategy


@pytest.fixture
def strategy() -> MicroStructureStrategy:
    return MicroStructureStrategy()


def _make_ohlcv(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    """Build an OHLCV DataFrame from explicit close (and optional high/low) arrays."""
    n = len(closes)
    dates = pd.date_range("2024-06-03 09:30", periods=n, freq="5min", tz="US/Eastern")
    if highs is None:
        highs = [c + 1.0 for c in closes]
    if lows is None:
        lows = [c - 1.0 for c in closes]
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1000] * n,
        },
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


class TestMicroStructureProperties:
    """Strategy metadata and ABC compliance."""

    def test_is_base_strategy(self, strategy: MicroStructureStrategy) -> None:
        assert isinstance(strategy, BaseStrategy)

    def test_name_is_micro_structure(self, strategy: MicroStructureStrategy) -> None:
        assert strategy.name == "micro_structure"

    def test_instruments(self, strategy: MicroStructureStrategy) -> None:
        assert strategy.instruments == ["ES", "YM", "RTY", "GC"]

    def test_timeframes(self, strategy: MicroStructureStrategy) -> None:
        assert strategy.timeframes == ["5m"]


class TestMicroStructureSignals:
    """Signal generation logic."""

    def test_generate_signals_returns_list(
        self, strategy: MicroStructureStrategy
    ) -> None:
        # Flat data — no pattern expected, but return type must be list
        closes = [100.0] * 30
        data = _make_ohlcv(closes)
        result = strategy.generate_signals("ES", data)
        assert isinstance(result, list)

    def test_failed_breakout_above_generates_short(
        self, strategy: MicroStructureStrategy
    ) -> None:
        """Price breaks above the lookback high then closes back below -> SHORT."""
        # 20 bars of baseline around 100, high never exceeds 105
        closes = [100.0] * 20
        highs = [105.0] * 20
        lows = [95.0] * 20

        # Bar 21: price spikes above the 105 high (breakout bar)
        closes.append(106.0)
        highs.append(107.0)
        lows.append(104.0)

        # Bars 22-23 (confirmation_bars=2): price falls back below the 105 high
        for _ in range(2):
            closes.append(103.0)
            highs.append(104.0)
            lows.append(102.0)

        data = _make_ohlcv(closes, highs=highs, lows=lows)
        signals = strategy.generate_signals("ES", data)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == "SHORT"
        assert sig.strategy == "micro_structure"
        assert sig.symbol == "ES"
        assert sig.micro_symbol == "MES"
        assert sig.timeframe == "5m"

    def test_failed_breakdown_below_generates_long(
        self, strategy: MicroStructureStrategy
    ) -> None:
        """Price breaks below the lookback low then closes back above -> LONG."""
        closes = [100.0] * 20
        highs = [105.0] * 20
        lows = [95.0] * 20

        # Bar 21: price dips below the 95 low (breakdown bar)
        closes.append(94.0)
        highs.append(96.0)
        lows.append(93.0)

        # Bars 22-23: price recovers back above the 95 low
        for _ in range(2):
            closes.append(97.0)
            highs.append(98.0)
            lows.append(96.0)

        data = _make_ohlcv(closes, highs=highs, lows=lows)
        signals = strategy.generate_signals("ES", data)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.direction == "LONG"
        assert sig.strategy == "micro_structure"
        assert sig.symbol == "ES"
        assert sig.micro_symbol == "MES"

    def test_no_pattern_returns_empty(
        self, strategy: MicroStructureStrategy
    ) -> None:
        """Steadily trending data with no failed breakout should yield no signals."""
        n = 30
        closes = [100.0 + i * 0.5 for i in range(n)]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        data = _make_ohlcv(closes, highs=highs, lows=lows)
        signals = strategy.generate_signals("ES", data)
        assert signals == []

    def test_insufficient_data_returns_empty(
        self, strategy: MicroStructureStrategy
    ) -> None:
        """Fewer bars than lookback + confirmation_bars should return empty."""
        closes = [100.0] * 5
        data = _make_ohlcv(closes)
        signals = strategy.generate_signals("ES", data)
        assert signals == []

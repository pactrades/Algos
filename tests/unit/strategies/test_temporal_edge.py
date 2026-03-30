"""Tests for TemporalEdge strategy — time-based structural patterns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algos.core.signal import Signal
from algos.strategies.base import BaseStrategy
from algos.strategies.temporal_edge import TemporalEdgeStrategy


@pytest.fixture
def strategy() -> TemporalEdgeStrategy:
    return TemporalEdgeStrategy()


@pytest.fixture
def sufficient_data() -> pd.DataFrame:
    """Synthetic 5-min data spanning 65 days with a strong bullish bias at hour 14 UTC.

    At hour 14, closes are consistently higher than opens (bullish).
    All other hours have random noise centered around zero.
    """
    np.random.seed(42)
    days = 65
    # Build 5-min bars for each day, 24 hours * 12 bars/hour = 288 bars/day
    timestamps = pd.date_range(
        "2024-01-01", periods=days * 288, freq="5min", tz="UTC"
    )
    n = len(timestamps)
    base_price = 5000.0
    closes = np.full(n, base_price)
    opens = np.full(n, base_price)

    for i, ts in enumerate(timestamps):
        noise = np.random.randn() * 0.5
        if ts.hour == 14:
            # Strong bullish bias: close always above open
            opens[i] = base_price + noise
            closes[i] = base_price + 3.0 + abs(noise)
        else:
            opens[i] = base_price + noise
            closes[i] = base_price + noise + np.random.randn() * 0.3

    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n)) * 0.5
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n)) * 0.5
    volume = np.random.randint(100, 5000, n)

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )
    return df


@pytest.fixture
def insufficient_data() -> pd.DataFrame:
    """Only 10 bars — far too few for any bucket to reach min_bars_per_bucket."""
    np.random.seed(0)
    timestamps = pd.date_range("2024-01-01 10:00", periods=10, freq="5min", tz="UTC")
    n = len(timestamps)
    close = 5000.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.3,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.random.randint(100, 5000, n),
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )


class TestTemporalEdgeProperties:
    """Test basic strategy properties and type."""

    def test_is_base_strategy(self, strategy: TemporalEdgeStrategy) -> None:
        assert isinstance(strategy, BaseStrategy)

    def test_name_is_temporal_edge(self, strategy: TemporalEdgeStrategy) -> None:
        assert strategy.name == "temporal_edge"

    def test_instruments(self, strategy: TemporalEdgeStrategy) -> None:
        assert strategy.instruments == ["ES", "RTY", "CL", "GC"]

    def test_timeframes(self, strategy: TemporalEdgeStrategy) -> None:
        assert strategy.timeframes == ["5m"]


class TestTemporalEdgeSignals:
    """Test signal generation logic."""

    def test_generate_signals_returns_list(
        self, strategy: TemporalEdgeStrategy, sufficient_data: pd.DataFrame
    ) -> None:
        result = strategy.generate_signals("ES", sufficient_data)
        assert isinstance(result, list)

    def test_generate_signals_with_sufficient_data(
        self, sufficient_data: pd.DataFrame
    ) -> None:
        """Data has a strong bullish pattern at hour 14.

        Set the last bar to be at hour 14 so a signal is emitted.
        """
        # Trim data so the last timestamp falls in hour 14
        hour_14_mask = sufficient_data.index.hour == 14
        last_hour_14_ts = sufficient_data.index[hour_14_mask][-1]
        trimmed = sufficient_data.loc[:last_hour_14_ts]

        strat = TemporalEdgeStrategy()
        signals = strat.generate_signals("ES", trimmed)
        assert len(signals) >= 1
        assert all(isinstance(s, Signal) for s in signals)

    def test_generate_signals_with_insufficient_data(
        self, strategy: TemporalEdgeStrategy, insufficient_data: pd.DataFrame
    ) -> None:
        signals = strategy.generate_signals("ES", insufficient_data)
        assert signals == []

    def test_signal_has_correct_fields(self, sufficient_data: pd.DataFrame) -> None:
        """Verify signal fields are correctly populated."""
        hour_14_mask = sufficient_data.index.hour == 14
        last_hour_14_ts = sufficient_data.index[hour_14_mask][-1]
        trimmed = sufficient_data.loc[:last_hour_14_ts]

        strat = TemporalEdgeStrategy()
        signals = strat.generate_signals("ES", trimmed)
        assert len(signals) >= 1

        sig = signals[0]
        assert sig.strategy == "temporal_edge"
        assert sig.symbol == "ES"
        assert sig.micro_symbol == "MES"
        assert sig.direction in ("LONG", "SHORT")
        assert sig.timeframe == "5m"
        assert sig.entry_price > 0
        assert sig.contracts >= 1

        # For a LONG signal, stop_loss < entry < take_profit
        if sig.direction == "LONG":
            assert sig.stop_loss < sig.entry_price
            if sig.take_profit is not None:
                assert sig.take_profit > sig.entry_price
        else:
            assert sig.stop_loss > sig.entry_price
            if sig.take_profit is not None:
                assert sig.take_profit < sig.entry_price

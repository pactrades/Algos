"""Tests for CalendarAlpha strategy — seasonal and calendar effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algos.core.signal import Signal
from algos.strategies.base import BaseStrategy
from algos.strategies.calendar_alpha import CalendarAlphaStrategy


@pytest.fixture
def strategy() -> CalendarAlphaStrategy:
    return CalendarAlphaStrategy()


@pytest.fixture
def monday_bullish_data() -> pd.DataFrame:
    """Synthetic daily data spanning ~30 weeks where Mondays always go up.

    Mondays have a consistent positive return; other days are flat/noisy.
    The last row lands on a Monday so a signal should fire.
    """
    np.random.seed(42)
    # Start on a Monday and generate ~150 trading days (business days)
    timestamps = pd.bdate_range("2024-01-01", periods=150, tz="UTC")
    n = len(timestamps)
    base_price = 70.0
    opens = np.full(n, base_price)
    closes = np.full(n, base_price)

    for i, ts in enumerate(timestamps):
        noise = np.random.randn() * 0.02
        if ts.dayofweek == 0:  # Monday
            # Strong positive return
            opens[i] = base_price + noise
            closes[i] = base_price + 0.5 + abs(noise)
        else:
            # Flat / tiny noise
            opens[i] = base_price + noise
            closes[i] = base_price + noise * 0.1

    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n)) * 0.1
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n)) * 0.1
    volume = np.random.randint(1000, 50000, n)

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )
    # Ensure last row is a Monday
    monday_mask = df.index.dayofweek == 0
    last_monday = df.index[monday_mask][-1]
    return df.loc[:last_monday]


@pytest.fixture
def no_effect_data() -> pd.DataFrame:
    """Random daily data with no day-of-week bias."""
    np.random.seed(99)
    timestamps = pd.bdate_range("2024-01-01", periods=150, tz="UTC")
    n = len(timestamps)
    base_price = 5000.0
    noise = np.random.randn(n) * 0.0001  # tiny symmetric noise
    opens = np.full(n, base_price) + noise
    closes = np.full(n, base_price) + noise * 0.5

    highs = np.maximum(opens, closes) + 0.01
    lows = np.minimum(opens, closes) - 0.01
    volume = np.random.randint(1000, 50000, n)

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )


@pytest.fixture
def insufficient_data() -> pd.DataFrame:
    """Only 10 bars — too few for any day to reach min_samples_per_day."""
    np.random.seed(0)
    timestamps = pd.bdate_range("2024-01-01", periods=10, tz="UTC")
    n = len(timestamps)
    close = 70.0 + np.cumsum(np.random.randn(n) * 0.1)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.05,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.random.randint(1000, 50000, n),
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )


class TestCalendarAlphaProperties:
    """Test basic strategy properties and type."""

    def test_is_base_strategy(self, strategy: CalendarAlphaStrategy) -> None:
        assert isinstance(strategy, BaseStrategy)

    def test_name_is_calendar_alpha(self, strategy: CalendarAlphaStrategy) -> None:
        assert strategy.name == "calendar_alpha"

    def test_instruments(self, strategy: CalendarAlphaStrategy) -> None:
        assert strategy.instruments == ["CL", "GC", "ZB", "ES"]

    def test_timeframes(self, strategy: CalendarAlphaStrategy) -> None:
        assert strategy.timeframes == ["1d"]


class TestCalendarAlphaSignals:
    """Test signal generation logic."""

    def test_generate_signals_returns_list(
        self, strategy: CalendarAlphaStrategy, monday_bullish_data: pd.DataFrame
    ) -> None:
        result = strategy.generate_signals("CL", monday_bullish_data)
        assert isinstance(result, list)

    def test_generate_signals_with_day_effect(self, monday_bullish_data: pd.DataFrame) -> None:
        """Data has a strong bullish Monday pattern and ends on Monday."""
        strat = CalendarAlphaStrategy()
        signals = strat.generate_signals("CL", monday_bullish_data)
        assert len(signals) >= 1
        assert all(isinstance(s, Signal) for s in signals)

    def test_generate_signals_no_effect(
        self, strategy: CalendarAlphaStrategy, no_effect_data: pd.DataFrame
    ) -> None:
        """Random symmetric data should produce no signal."""
        signals = strategy.generate_signals("ES", no_effect_data)
        assert signals == []

    def test_generate_signals_insufficient_data(
        self, strategy: CalendarAlphaStrategy, insufficient_data: pd.DataFrame
    ) -> None:
        signals = strategy.generate_signals("CL", insufficient_data)
        assert signals == []

    def test_signal_direction_matches_historical_bias(
        self, monday_bullish_data: pd.DataFrame
    ) -> None:
        """Positive historical Monday bias should produce a LONG signal."""
        strat = CalendarAlphaStrategy()
        signals = strat.generate_signals("CL", monday_bullish_data)
        assert len(signals) >= 1
        assert signals[0].direction == "LONG"
        assert signals[0].strategy == "calendar_alpha"
        assert signals[0].symbol == "CL"
        assert signals[0].micro_symbol == "MCL"
        assert signals[0].timeframe == "1d"

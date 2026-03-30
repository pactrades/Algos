"""Tests for the VolRegime volatility compression strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algos.core.signal import Signal
from algos.strategies.base import BaseStrategy
from algos.strategies.vol_regime import VolRegimeStrategy


@pytest.fixture
def strategy() -> VolRegimeStrategy:
    return VolRegimeStrategy()


def _make_ohlcv(
    n: int,
    base: float = 100.0,
    volatility: float = 2.0,
    *,
    squeeze_last: int = 0,
    squeeze_vol: float = 0.05,
    bullish_last: bool = True,
) -> pd.DataFrame:
    """Build synthetic OHLCV data.

    Args:
        n: Number of bars.
        base: Starting price.
        volatility: Normal bar range (high-low spread).
        squeeze_last: Number of trailing bars with compressed volatility.
        squeeze_vol: Volatility multiplier for squeeze bars (very small).
        bullish_last: If True the last bar closes above open, else below.
    """
    np.random.seed(42)
    dates = pd.date_range("2024-01-02 09:00", periods=n, freq="1h", tz="US/Eastern")

    close = base + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n)) * volatility
    low = close - np.abs(np.random.randn(n)) * volatility
    open_ = close + np.random.randn(n) * 0.3

    # Compress volatility on the last `squeeze_last` bars
    if squeeze_last > 0:
        for i in range(n - squeeze_last, n):
            mid = close[i]
            high[i] = mid + squeeze_vol
            low[i] = mid - squeeze_vol
            # Force direction on the very last bar
            if i == n - 1:
                if bullish_last:
                    open_[i] = mid - squeeze_vol * 0.5
                    close[i] = mid + squeeze_vol * 0.5
                else:
                    open_[i] = mid + squeeze_vol * 0.5
                    close[i] = mid - squeeze_vol * 0.5

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(100, 5000, n),
        },
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


class TestVolRegimeProperties:
    """Test static strategy properties."""

    def test_is_base_strategy(self, strategy: VolRegimeStrategy) -> None:
        assert isinstance(strategy, BaseStrategy)

    def test_name_is_vol_regime(self, strategy: VolRegimeStrategy) -> None:
        assert strategy.name == "vol_regime"

    def test_instruments(self, strategy: VolRegimeStrategy) -> None:
        assert strategy.instruments == ["ES", "CL", "GC"]

    def test_timeframes(self, strategy: VolRegimeStrategy) -> None:
        assert strategy.timeframes == ["1h"]


class TestVolRegimeSignals:
    """Test signal generation logic."""

    def test_generate_signals_returns_list(self, strategy: VolRegimeStrategy) -> None:
        data = _make_ohlcv(100)
        result = strategy.generate_signals("ES", data)
        assert isinstance(result, list)

    def test_generate_signals_during_squeeze(self, strategy: VolRegimeStrategy) -> None:
        data = _make_ohlcv(100, squeeze_last=20, squeeze_vol=0.01)
        signals = strategy.generate_signals("ES", data)
        assert len(signals) == 1
        sig = signals[0]
        assert isinstance(sig, Signal)
        assert sig.strategy == "vol_regime"
        assert sig.symbol == "ES"
        assert sig.micro_symbol == "MES"
        assert sig.timeframe == "1h"

    def test_generate_signals_no_squeeze(self, strategy: VolRegimeStrategy) -> None:
        # All bars have uniform high volatility — no squeeze detected.
        # We build constant-range bars so every ATR value is identical,
        # meaning the last ATR can never be below any percentile.
        np.random.seed(0)
        n = 100
        dates = pd.date_range("2024-01-02 09:00", periods=n, freq="1h", tz="US/Eastern")
        base = 100.0 + np.cumsum(np.random.randn(n) * 0.1)
        spread = 5.0  # constant large spread on every bar
        data = pd.DataFrame(
            {
                "open": base,
                "high": base + spread,
                "low": base - spread,
                "close": base + 0.1,
                "volume": np.full(n, 1000),
            },
            index=pd.DatetimeIndex(dates, name="timestamp"),
        )
        signals = strategy.generate_signals("ES", data)
        assert signals == []

    def test_generate_signals_insufficient_data(self, strategy: VolRegimeStrategy) -> None:
        # Fewer bars than atr_period + lookback
        data = _make_ohlcv(10)
        signals = strategy.generate_signals("ES", data)
        assert signals == []

    def test_signal_direction_follows_last_bar(self, strategy: VolRegimeStrategy) -> None:
        # Bullish last bar → LONG
        data_bull = _make_ohlcv(100, squeeze_last=20, squeeze_vol=0.01, bullish_last=True)
        signals_bull = strategy.generate_signals("ES", data_bull)
        assert len(signals_bull) == 1
        assert signals_bull[0].direction == "LONG"

        # Bearish last bar → SHORT
        data_bear = _make_ohlcv(100, squeeze_last=20, squeeze_vol=0.01, bullish_last=False)
        signals_bear = strategy.generate_signals("ES", data_bear)
        assert len(signals_bear) == 1
        assert signals_bear[0].direction == "SHORT"

"""Tests for the InterMarketFlow strategy — cross-instrument lead/lag."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algos.core.signal import Signal
from algos.strategies.base import BaseStrategy
from algos.strategies.intermarket_flow import InterMarketFlow


@pytest.fixture
def strategy() -> InterMarketFlow:
    return InterMarketFlow()


@pytest.fixture
def _dates() -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02 09:30", periods=100, freq="1h", tz="US/Eastern")


def _make_ohlcv(
    dates: pd.DatetimeIndex,
    base: float = 5000.0,
    *,
    spike_at: int | None = None,
    spike_size: float = 0.0,
) -> pd.DataFrame:
    """Build synthetic OHLCV data with an optional price spike."""
    np.random.seed(42)
    n = len(dates)
    returns = np.random.randn(n) * 0.001  # small normal returns
    if spike_at is not None:
        returns[spike_at] = spike_size
    close = base * np.cumprod(1 + returns)
    high = close * 1.001
    low = close * 0.999
    open_ = close * (1 + np.random.randn(n) * 0.0002)
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


class TestInterMarketFlowProperties:
    """Test strategy metadata / interface compliance."""

    def test_is_base_strategy(self, strategy: InterMarketFlow) -> None:
        assert isinstance(strategy, BaseStrategy)

    def test_name_is_intermarket_flow(self, strategy: InterMarketFlow) -> None:
        assert strategy.name == "intermarket_flow"

    def test_instruments(self, strategy: InterMarketFlow) -> None:
        assert strategy.instruments == ["ES", "ZB"]

    def test_timeframes(self, strategy: InterMarketFlow) -> None:
        assert strategy.timeframes == ["1h"]


class TestInterMarketFlowSignals:
    """Test signal generation logic."""

    def test_generate_signals_returns_list(
        self, strategy: InterMarketFlow, _dates: pd.DatetimeIndex
    ) -> None:
        data = _make_ohlcv(_dates)
        result = strategy.generate_signals("ES", data)
        assert isinstance(result, list)

    def test_generate_signals_with_extreme_move(
        self, strategy: InterMarketFlow, _dates: pd.DatetimeIndex
    ) -> None:
        """A sharp spike should trigger a mean-reversion signal."""
        data = _make_ohlcv(_dates, spike_at=99, spike_size=0.05)
        signals = strategy.generate_signals("ES", data)
        assert len(signals) >= 1
        assert all(isinstance(s, Signal) for s in signals)
        assert signals[0].strategy == "intermarket_flow"
        assert signals[0].symbol == "ES"
        assert signals[0].micro_symbol == "MES"

    def test_generate_signals_with_normal_data(
        self, strategy: InterMarketFlow, _dates: pd.DatetimeIndex
    ) -> None:
        """Normal, calm data should produce no signals."""
        data = _make_ohlcv(_dates)
        signals = strategy.generate_signals("ES", data)
        assert signals == []

    def test_generate_signals_insufficient_data(self, strategy: InterMarketFlow) -> None:
        """Fewer bars than lookback_period → empty list."""
        dates = pd.date_range("2024-01-02 09:30", periods=5, freq="1h", tz="US/Eastern")
        data = _make_ohlcv(dates)
        signals = strategy.generate_signals("ES", data)
        assert signals == []

    def test_signal_direction_correct_sharp_up(
        self, strategy: InterMarketFlow, _dates: pd.DatetimeIndex
    ) -> None:
        """Sharp UP move → SHORT mean-reversion signal."""
        data = _make_ohlcv(_dates, spike_at=99, spike_size=0.05)
        signals = strategy.generate_signals("ES", data)
        assert len(signals) >= 1
        assert signals[0].direction == "SHORT"

    def test_signal_direction_correct_sharp_down(
        self, strategy: InterMarketFlow, _dates: pd.DatetimeIndex
    ) -> None:
        """Sharp DOWN move → LONG mean-reversion signal."""
        data = _make_ohlcv(_dates, spike_at=99, spike_size=-0.05)
        signals = strategy.generate_signals("ES", data)
        assert len(signals) >= 1
        assert signals[0].direction == "LONG"

    def test_zb_micro_symbol(self, strategy: InterMarketFlow, _dates: pd.DatetimeIndex) -> None:
        """ZB has no micro contract — micro_symbol should stay ZB."""
        data = _make_ohlcv(_dates, base=120.0, spike_at=99, spike_size=0.05)
        signals = strategy.generate_signals("ZB", data)
        assert len(signals) >= 1
        assert signals[0].micro_symbol == "ZB"

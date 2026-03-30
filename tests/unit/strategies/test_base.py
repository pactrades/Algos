"""Tests for base strategy abstract class and strategy runner."""

import numpy as np
import pandas as pd
import pytest

from algos.core.signal import Signal
from algos.strategies.base import BaseStrategy, StrategyRunner


class DummyStrategy(BaseStrategy):
    """Concrete strategy for testing the base class."""

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def instruments(self) -> list[str]:
        return ["ES", "CL"]

    @property
    def timeframes(self) -> list[str]:
        return ["5m"]

    def generate_signals(self, symbol: str, data: pd.DataFrame) -> list[Signal]:
        """Always generate one LONG signal if we have data."""
        if len(data) < 2:
            return []
        last = data.iloc[-1]
        return [
            Signal(
                strategy=self.name,
                symbol=symbol,
                micro_symbol="MES" if symbol == "ES" else "MCL",
                direction="LONG",
                entry_price=float(last["close"]),
                stop_loss=float(last["close"] - 10),
                take_profit=float(last["close"] + 20),
                contracts=1,
                risk_dollars=12.50,
                timeframe="5m",
                timestamp=data.index[-1].to_pydatetime(),
                confidence=0.8,
            )
        ]


class NeverSignalStrategy(BaseStrategy):
    """Strategy that never generates signals."""

    @property
    def name(self) -> str:
        return "never_signal"

    @property
    def instruments(self) -> list[str]:
        return ["GC"]

    @property
    def timeframes(self) -> list[str]:
        return ["1h"]

    def generate_signals(self, symbol: str, data: pd.DataFrame) -> list[Signal]:
        return []


@pytest.fixture
def sample_data() -> dict[str, pd.DataFrame]:
    """Multi-instrument OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-02 09:30", periods=50, freq="5min", tz="US/Eastern")

    def make_df(base: float) -> pd.DataFrame:
        n = 50
        close = base + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_ = close + np.random.randn(n) * 0.3
        return pd.DataFrame(
            {
                "open": open_, "high": high, "low": low,
                "close": close, "volume": np.random.randint(100, 5000, n),
            },
            index=pd.DatetimeIndex(dates, name="timestamp"),
        )

    return {"ES": make_df(5000.0), "CL": make_df(75.0), "GC": make_df(2050.0)}


class TestBaseStrategy:
    """Test the abstract base strategy interface."""

    def test_dummy_strategy_has_name(self) -> None:
        strat = DummyStrategy()
        assert strat.name == "dummy"

    def test_dummy_strategy_has_instruments(self) -> None:
        strat = DummyStrategy()
        assert strat.instruments == ["ES", "CL"]

    def test_generate_signals_returns_list(self, sample_data: dict) -> None:
        strat = DummyStrategy()
        signals = strat.generate_signals("ES", sample_data["ES"])
        assert isinstance(signals, list)
        assert len(signals) == 1
        assert isinstance(signals[0], Signal)

    def test_signal_has_correct_strategy_name(self, sample_data: dict) -> None:
        strat = DummyStrategy()
        signals = strat.generate_signals("ES", sample_data["ES"])
        assert signals[0].strategy == "dummy"

    def test_never_signal_returns_empty(self, sample_data: dict) -> None:
        strat = NeverSignalStrategy()
        signals = strat.generate_signals("GC", sample_data["GC"])
        assert signals == []


class TestStrategyRunner:
    """Test the strategy runner/orchestrator."""

    def test_runner_collects_signals(self, sample_data: dict) -> None:
        runner = StrategyRunner(strategies=[DummyStrategy()])
        signals = runner.run(sample_data)
        # DummyStrategy trades ES and CL, should get 2 signals
        assert len(signals) == 2

    def test_runner_with_multiple_strategies(self, sample_data: dict) -> None:
        runner = StrategyRunner(strategies=[DummyStrategy(), NeverSignalStrategy()])
        signals = runner.run(sample_data)
        # DummyStrategy: 2 signals (ES, CL). NeverSignal: 0. Total: 2
        assert len(signals) == 2

    def test_runner_with_no_strategies(self, sample_data: dict) -> None:
        runner = StrategyRunner(strategies=[])
        signals = runner.run(sample_data)
        assert signals == []

    def test_runner_skips_missing_instruments(self, sample_data: dict) -> None:
        """If strategy wants RTY but data doesn't have it, skip gracefully."""
        strat = DummyStrategy()  # Wants ES, CL
        limited_data = {"ES": sample_data["ES"]}  # Only ES available
        runner = StrategyRunner(strategies=[strat])
        signals = runner.run(limited_data)
        assert len(signals) == 1  # Only ES signal

"""Integration tests for StrategyRunner with all five strategies."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from algos.core.signal import Signal
from algos.strategies.base import StrategyRunner
from algos.strategies.calendar_alpha import CalendarAlphaStrategy
from algos.strategies.intermarket_flow import InterMarketFlow
from algos.strategies.micro_structure import MicroStructureStrategy
from algos.strategies.temporal_edge import TemporalEdgeStrategy
from algos.strategies.vol_regime import VolRegimeStrategy


def _make_ohlcv_5m(
    bars: int = 200,
    base_price: float = 5000.0,
    start: datetime | None = None,
) -> pd.DataFrame:
    """Generate synthetic 5-minute OHLCV data."""
    if start is None:
        start = datetime(2025, 1, 6, 9, 0, tzinfo=UTC)
    rng = np.random.default_rng(42)
    timestamps = pd.date_range(start=start, periods=bars, freq="5min", tz=UTC)
    close = base_price + np.cumsum(rng.normal(0, 0.5, bars))
    high = close + rng.uniform(0.2, 1.0, bars)
    low = close - rng.uniform(0.2, 1.0, bars)
    open_ = close + rng.normal(0, 0.3, bars)
    volume = rng.integers(100, 5000, bars)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=timestamps,
    )


def _make_ohlcv_1h(
    bars: int = 200,
    base_price: float = 5000.0,
    start: datetime | None = None,
) -> pd.DataFrame:
    """Generate synthetic 1-hour OHLCV data."""
    if start is None:
        start = datetime(2025, 1, 6, 9, 0, tzinfo=UTC)
    rng = np.random.default_rng(99)
    timestamps = pd.date_range(start=start, periods=bars, freq="1h", tz=UTC)
    close = base_price + np.cumsum(rng.normal(0, 1.0, bars))
    high = close + rng.uniform(0.5, 2.0, bars)
    low = close - rng.uniform(0.5, 2.0, bars)
    open_ = close + rng.normal(0, 0.5, bars)
    volume = rng.integers(500, 10000, bars)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=timestamps,
    )


def _make_ohlcv_1d(
    bars: int = 200,
    base_price: float = 70.0,
    start: datetime | None = None,
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV data."""
    if start is None:
        start = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    rng = np.random.default_rng(7)
    timestamps = pd.date_range(start=start, periods=bars, freq="B", tz=UTC)
    close = base_price + np.cumsum(rng.normal(0, 0.5, bars))
    high = close + rng.uniform(0.3, 1.5, bars)
    low = close - rng.uniform(0.3, 1.5, bars)
    open_ = close + rng.normal(0, 0.3, bars)
    volume = rng.integers(1000, 50000, bars)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=timestamps,
    )


def _build_multi_instrument_data() -> dict[str, pd.DataFrame]:
    """Build synthetic data covering all instruments used by the 5 strategies."""
    return {
        "ES": _make_ohlcv_5m(bars=300, base_price=5200.0),
        "CL": _make_ohlcv_5m(bars=300, base_price=75.0),
        "GC": _make_ohlcv_5m(bars=300, base_price=2050.0),
        "ZB": _make_ohlcv_1h(bars=300, base_price=118.0),
        "YM": _make_ohlcv_5m(bars=300, base_price=38000.0),
        "RTY": _make_ohlcv_5m(bars=300, base_price=2000.0),
    }


def _build_all_strategies() -> list:
    """Instantiate all five strategies with default parameters."""
    return [
        TemporalEdgeStrategy(min_bars_per_bucket=10, significance_threshold=0.55),
        InterMarketFlow(lookback_period=20, z_score_threshold=1.0),
        VolRegimeStrategy(atr_period=14, squeeze_percentile=30, lookback=50),
        MicroStructureStrategy(lookback=20, confirmation_bars=2),
        CalendarAlphaStrategy(min_samples_per_day=10, effect_threshold=0.0005),
    ]


class TestAllFiveStrategiesRunTogether:
    """Verify StrategyRunner can execute all 5 strategies against multi-instrument data."""

    def test_all_five_strategies_run_together(self) -> None:
        strategies = _build_all_strategies()
        runner = StrategyRunner(strategies=strategies)
        data = _build_multi_instrument_data()

        signals = runner.run(data)

        assert isinstance(signals, list)
        for sig in signals:
            assert isinstance(sig, Signal)
            assert sig.entry_price > 0
            assert sig.risk_dollars > 0
            assert sig.direction in ("LONG", "SHORT")
            assert sig.strategy in {s.name for s in strategies}


class TestRunnerWithRealisticDataVolume:
    """Verify all strategies handle 500+ bars without errors."""

    def test_runner_with_realistic_data_volume(self) -> None:
        start_5m = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)
        start_1h = datetime(2024, 6, 1, 9, 0, tzinfo=UTC)

        data: dict[str, pd.DataFrame] = {
            "ES": _make_ohlcv_5m(bars=600, base_price=5200.0, start=start_5m),
            "CL": _make_ohlcv_5m(bars=600, base_price=75.0, start=start_5m),
            "GC": _make_ohlcv_5m(bars=600, base_price=2050.0, start=start_5m),
            "ZB": _make_ohlcv_1h(bars=600, base_price=118.0, start=start_1h),
            "YM": _make_ohlcv_5m(bars=600, base_price=38000.0, start=start_5m),
            "RTY": _make_ohlcv_5m(bars=600, base_price=2000.0, start=start_5m),
        }

        strategies = _build_all_strategies()
        runner = StrategyRunner(strategies=strategies)

        signals = runner.run(data)

        assert isinstance(signals, list)
        for sig in signals:
            assert isinstance(sig, Signal)


class TestStrategiesHaveUniqueNames:
    """Verify no two strategies share a name."""

    def test_strategies_have_unique_names(self) -> None:
        strategies = _build_all_strategies()
        names = [s.name for s in strategies]
        assert len(names) == len(set(names)), f"Duplicate strategy names found: {names}"
        assert len(names) == 5

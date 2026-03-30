"""Integration tests for the signal pipeline: strategy -> risk gate -> output."""

from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO
from unittest.mock import patch

import numpy as np
import pandas as pd

from algos.core.config import PropFirmProfile
from algos.core.risk import RiskGate
from algos.core.signal import Signal
from algos.output.console import ConsoleOutputHandler
from algos.output.dispatcher import OutputDispatcher
from algos.strategies.base import StrategyRunner
from algos.strategies.intermarket_flow import InterMarketFlow
from algos.strategies.vol_regime import VolRegimeStrategy


def _default_profile() -> PropFirmProfile:
    return PropFirmProfile(
        name="test_firm",
        trailing_drawdown=2500.0,
        daily_loss_limit=1000.0,
        profit_target=6000.0,
        max_contracts=10,
        account_size=50000.0,
    )


def _make_signal(
    symbol: str = "ES",
    direction: str = "LONG",
    entry: float = 5200.0,
    risk: float = 50.0,
    timestamp: datetime | None = None,
    strategy: str = "temporal_edge",
) -> Signal:
    """Build a valid Signal for testing."""
    if timestamp is None:
        timestamp = datetime(2025, 3, 15, 14, 0, tzinfo=UTC)

    if direction == "LONG":
        stop_loss = entry - risk
        take_profit = entry + risk * 1.5
    else:
        stop_loss = entry + risk
        take_profit = entry - risk * 1.5

    return Signal(
        strategy=strategy,
        symbol=symbol,
        micro_symbol=f"M{symbol}",
        direction=direction,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        contracts=1,
        risk_dollars=risk,
        timeframe="5m",
        timestamp=timestamp,
        confidence=0.65,
    )


def _make_ohlcv_1h(
    bars: int = 200,
    base_price: float = 5000.0,
    start: datetime | None = None,
) -> pd.DataFrame:
    if start is None:
        start = datetime(2025, 1, 6, 9, 0, tzinfo=UTC)
    rng = np.random.default_rng(42)
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


class TestSignalThroughRiskGate:
    """Generate signals and pass each through RiskGate."""

    def test_signal_through_risk_gate(self) -> None:
        profile = _default_profile()
        gate = RiskGate(profile)

        signals = [
            _make_signal(symbol="ES", risk=50.0),
            _make_signal(symbol="CL", entry=75.0, risk=2.0),
            _make_signal(symbol="GC", entry=2050.0, risk=10.0, direction="SHORT"),
        ]

        approved = [s for s in signals if gate.approve(s)]

        assert len(approved) > 0
        for sig in approved:
            assert isinstance(sig, Signal)
            assert sig.entry_price > 0
            assert sig.risk_dollars > 0


class TestSignalToOutput:
    """Generate signals and emit through ConsoleOutputHandler."""

    def test_signal_to_output(self) -> None:
        handler = ConsoleOutputHandler(json_output=False)
        signal = _make_signal()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            handler.emit(signal)
            output = mock_stdout.getvalue()

        assert "ES" in output
        assert "LONG" in output
        assert "5200" in output
        assert "temporal_edge" in output

    def test_signal_to_json_output(self) -> None:
        handler = ConsoleOutputHandler(json_output=True)
        signal = _make_signal()

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            handler.emit(signal)
            output = mock_stdout.getvalue()

        assert '"symbol":"ES"' in output or '"symbol": "ES"' in output


class TestFullPipeline:
    """End-to-end: StrategyRunner -> RiskGate -> OutputDispatcher."""

    def test_full_pipeline(self) -> None:
        # Set up strategies
        strategies = [
            InterMarketFlow(lookback_period=20, z_score_threshold=0.5),
            VolRegimeStrategy(atr_period=14, squeeze_percentile=40, lookback=50),
        ]
        runner = StrategyRunner(strategies=strategies)

        data = {
            "ES": _make_ohlcv_1h(bars=200, base_price=5200.0),
            "ZB": _make_ohlcv_1h(bars=200, base_price=118.0),
            "CL": _make_ohlcv_1h(bars=200, base_price=75.0),
            "GC": _make_ohlcv_1h(bars=200, base_price=2050.0),
        }

        # Generate signals
        signals = runner.run(data)

        # Filter through risk gate
        profile = _default_profile()
        gate = RiskGate(profile)
        approved = [s for s in signals if gate.approve(s)]

        # Dispatch to output
        console = ConsoleOutputHandler(json_output=False)
        dispatcher = OutputDispatcher(handlers=[console])

        with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
            for sig in approved:
                dispatcher.emit(sig)
            output = mock_stdout.getvalue()

        # Pipeline ran without errors; if signals were generated they were emitted
        assert isinstance(approved, list)
        if approved:
            assert len(output) > 0


class TestNqSignalsBlockedDuringBlackout:
    """NQ signals at 10:00 AM ET must be blocked by RiskGate."""

    def test_nq_signals_blocked_during_blackout(self) -> None:
        profile = _default_profile()
        gate = RiskGate(profile)

        # 10:00 AM ET = 14:00 UTC (during blackout: 9:30-12:00 ET)
        blackout_ts = datetime(2025, 3, 15, 14, 0, tzinfo=UTC)
        nq_signal = _make_signal(
            symbol="NQ",
            entry=18500.0,
            risk=100.0,
            timestamp=blackout_ts,
        )

        assert gate.approve(nq_signal) is False

    def test_nq_signals_allowed_outside_blackout(self) -> None:
        profile = _default_profile()
        gate = RiskGate(profile)

        # 1:00 PM ET = 17:00 UTC (outside blackout)
        ok_ts = datetime(2025, 3, 15, 17, 0, tzinfo=UTC)
        nq_signal = _make_signal(
            symbol="NQ",
            entry=18500.0,
            risk=100.0,
            timestamp=ok_ts,
        )

        assert gate.approve(nq_signal) is True

    def test_non_nq_allowed_during_blackout(self) -> None:
        profile = _default_profile()
        gate = RiskGate(profile)

        blackout_ts = datetime(2025, 3, 15, 14, 0, tzinfo=UTC)
        es_signal = _make_signal(symbol="ES", timestamp=blackout_ts)

        assert gate.approve(es_signal) is True

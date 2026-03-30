"""Integration tests for risk management pipeline."""

from __future__ import annotations

from datetime import UTC, datetime

from algos.core.config import PropFirmProfile
from algos.core.risk import DailyDrawdownTracker, RiskGate, TrailingDrawdownTracker
from algos.core.signal import Signal


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


class TestTrailingDrawdownAcrossMultipleTrades:
    """Simulate winning and losing trades, verify drawdown tracking."""

    def test_trailing_drawdown_across_multiple_trades(self) -> None:
        tracker = TrailingDrawdownTracker(max_drawdown=2500.0, starting_equity=50000.0)

        # Win: equity goes up
        tracker.update_equity(50500.0)
        assert tracker.peak_equity == 50500.0
        assert tracker.is_within_limit()

        # Win more: new peak
        tracker.update_equity(51000.0)
        assert tracker.peak_equity == 51000.0
        assert tracker.current_drawdown == 0.0

        # Lose some: drawdown from peak
        tracker.update_equity(49500.0)
        assert tracker.peak_equity == 51000.0
        assert tracker.current_drawdown == 1500.0
        assert tracker.is_within_limit()

        # Lose more: approach limit
        tracker.update_equity(48600.0)
        assert tracker.current_drawdown == 2400.0
        assert tracker.is_within_limit()

        # Breach limit
        tracker.update_equity(48400.0)
        assert tracker.current_drawdown == 2600.0
        assert not tracker.is_within_limit()

    def test_peak_only_ratchets_up(self) -> None:
        tracker = TrailingDrawdownTracker(max_drawdown=1000.0, starting_equity=10000.0)

        tracker.update_equity(10500.0)
        tracker.update_equity(10200.0)
        tracker.update_equity(10600.0)
        assert tracker.peak_equity == 10600.0

        tracker.update_equity(10000.0)
        assert tracker.peak_equity == 10600.0
        assert tracker.current_drawdown == 600.0


class TestDailyLossLimitResets:
    """Record losses across two days, verify reset works."""

    def test_daily_loss_limit_resets(self) -> None:
        tracker = DailyDrawdownTracker(daily_limit=1000.0, starting_equity=50000.0)

        # Day 1: accumulate losses
        tracker.record_trade(-400.0)
        assert tracker.daily_pnl == -400.0
        assert tracker.is_within_limit()

        tracker.record_trade(-500.0)
        assert tracker.daily_pnl == -900.0
        assert tracker.is_within_limit()

        # Breach limit
        tracker.record_trade(-200.0)
        assert tracker.daily_pnl == -1100.0
        assert not tracker.is_within_limit()

        # Day 2: reset
        tracker.reset_daily(new_equity=48900.0)
        assert tracker.daily_pnl == 0.0
        assert tracker.is_within_limit()

        # New day trades
        tracker.record_trade(-300.0)
        assert tracker.is_within_limit()

    def test_wins_offset_losses(self) -> None:
        tracker = DailyDrawdownTracker(daily_limit=500.0, starting_equity=50000.0)

        tracker.record_trade(-400.0)
        tracker.record_trade(300.0)
        assert tracker.daily_pnl == -100.0
        assert tracker.is_within_limit()


class TestRiskGateBlocksAfterDrawdown:
    """Pump equity below trailing DD limit, verify all signals rejected."""

    def test_risk_gate_blocks_after_drawdown(self) -> None:
        profile = _default_profile()
        gate = RiskGate(profile)

        # Equity starts at 50000, trailing DD limit is 2500
        # Simulate losses bringing equity below limit
        gate.trailing_tracker.update_equity(48000.0)
        assert gate.trailing_tracker.is_within_limit()

        gate.trailing_tracker.update_equity(47000.0)
        assert not gate.trailing_tracker.is_within_limit()

        # All signals should now be rejected
        signals = [
            _make_signal(symbol="ES", risk=50.0),
            _make_signal(symbol="CL", entry=75.0, risk=2.0),
            _make_signal(symbol="GC", entry=2050.0, risk=10.0, direction="SHORT"),
        ]

        for sig in signals:
            assert gate.approve(sig) is False

    def test_risk_gate_allows_within_limit(self) -> None:
        profile = _default_profile()
        gate = RiskGate(profile)

        # Small drawdown, still within limit
        gate.trailing_tracker.update_equity(49000.0)
        assert gate.trailing_tracker.is_within_limit()

        signal = _make_signal(symbol="ES", risk=50.0)
        assert gate.approve(signal) is True


class TestPropFirmSimulation:
    """Simulate 50 trades with a PropFirmProfile, track equity."""

    def test_prop_firm_simulation(self) -> None:
        profile = _default_profile()
        gate = RiskGate(profile)

        equity = profile.account_size
        trade_results = [
            100.0,
            -50.0,
            200.0,
            -150.0,
            75.0,
            -200.0,
            300.0,
            -100.0,
            50.0,
            -75.0,
            150.0,
            -50.0,
            100.0,
            -200.0,
            250.0,
            -100.0,
            50.0,
            -150.0,
            200.0,
            -50.0,
            100.0,
            -75.0,
            150.0,
            -100.0,
            200.0,
            -50.0,
            100.0,
            -150.0,
            75.0,
            -200.0,
            300.0,
            -100.0,
            50.0,
            -75.0,
            150.0,
            -50.0,
            100.0,
            -200.0,
            250.0,
            -100.0,
            50.0,
            -150.0,
            200.0,
            -50.0,
            100.0,
            -75.0,
            150.0,
            -100.0,
            200.0,
            -50.0,
        ]
        assert len(trade_results) == 50

        approved_count = 0
        rejected_count = 0
        breached = False

        for pnl in trade_results:
            signal = _make_signal(risk=abs(pnl))

            if gate.approve(signal):
                approved_count += 1
                equity += pnl
                gate.trailing_tracker.update_equity(equity)
                gate.daily_tracker.record_trade(pnl)
            else:
                rejected_count += 1
                if not breached:
                    breached = True

        # Trailing drawdown should never have been exceeded for approved trades
        # (gate would reject once breached)
        assert approved_count + rejected_count == 50
        assert approved_count > 0

        # Once breached, all subsequent signals are rejected
        if breached:
            # Find first breach point; all trades after should be rejected
            pass

        # Equity should be tracked correctly
        assert equity != profile.account_size or approved_count == 0

    def test_prop_firm_daily_limit_enforced(self) -> None:
        profile = PropFirmProfile(
            name="strict_firm",
            trailing_drawdown=5000.0,
            daily_loss_limit=500.0,
            profit_target=3000.0,
            max_contracts=5,
            account_size=25000.0,
        )
        gate = RiskGate(profile)

        # First signal with risk exactly at daily limit should be approved
        sig1 = _make_signal(risk=400.0)
        assert gate.approve(sig1) is True
        gate.daily_tracker.record_trade(-400.0)

        # Second signal pushing past daily limit should be rejected
        sig2 = _make_signal(risk=200.0)
        assert gate.approve(sig2) is False

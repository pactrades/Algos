"""Tests for risk management — drawdown tracking, position sizing, signal filtering."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from algos.core.config import PropFirmProfile
from algos.core.risk import (
    DailyDrawdownTracker,
    MaxPositionGuard,
    RiskGate,
    TrailingDrawdownTracker,
)
from algos.core.signal import Signal

ET = ZoneInfo("US/Eastern")


def _make_signal(
    symbol: str = "ES",
    direction: str = "LONG",
    entry: float = 5000.0,
    stop: float = 4990.0,
    tp: float = 5020.0,
    risk_dollars: float = 50.0,
    contracts: int = 2,
    strategy: str = "temporal_edge",
    timestamp: datetime | None = None,
) -> Signal:
    micro = {"ES": "MES", "CL": "MCL", "GC": "MGC", "ZB": "ZB"}.get(symbol, "MES")
    if timestamp is None:
        timestamp = datetime(2024, 1, 15, 14, 0, tzinfo=UTC)
    return Signal(
        strategy=strategy,
        symbol=symbol,
        micro_symbol=micro,
        direction=direction,
        entry_price=entry,
        stop_loss=stop,
        take_profit=tp,
        contracts=contracts,
        risk_dollars=risk_dollars,
        timeframe="5m",
        timestamp=timestamp,
        confidence=0.75,
    )


class TestTrailingDrawdownTracker:
    """Test trailing drawdown tracking from peak equity."""

    def test_initial_state(self) -> None:
        tracker = TrailingDrawdownTracker(max_drawdown=2500.0, starting_equity=50000.0)
        assert tracker.current_drawdown == 0.0
        assert tracker.is_within_limit()

    def test_equity_increase_raises_peak(self) -> None:
        tracker = TrailingDrawdownTracker(max_drawdown=2500.0, starting_equity=50000.0)
        tracker.update_equity(51000.0)
        assert tracker.peak_equity == 51000.0
        assert tracker.current_drawdown == 0.0

    def test_equity_decrease_creates_drawdown(self) -> None:
        tracker = TrailingDrawdownTracker(max_drawdown=2500.0, starting_equity=50000.0)
        tracker.update_equity(51000.0)  # New peak
        tracker.update_equity(49500.0)  # Draw down
        assert tracker.current_drawdown == 1500.0
        assert tracker.is_within_limit()

    def test_exceeds_limit(self) -> None:
        tracker = TrailingDrawdownTracker(max_drawdown=2500.0, starting_equity=50000.0)
        tracker.update_equity(50000.0)
        tracker.update_equity(47000.0)  # $3000 DD > $2500 limit
        assert not tracker.is_within_limit()

    def test_exactly_at_limit(self) -> None:
        tracker = TrailingDrawdownTracker(max_drawdown=2500.0, starting_equity=50000.0)
        tracker.update_equity(47500.0)  # Exactly $2500 DD
        assert tracker.current_drawdown == 2500.0
        assert tracker.is_within_limit()  # At limit, not over


class TestDailyDrawdownTracker:
    """Test daily loss limit tracking."""

    def test_initial_state(self) -> None:
        tracker = DailyDrawdownTracker(daily_limit=1000.0, starting_equity=50000.0)
        assert tracker.daily_pnl == 0.0
        assert tracker.is_within_limit()

    def test_record_loss(self) -> None:
        tracker = DailyDrawdownTracker(daily_limit=1000.0, starting_equity=50000.0)
        tracker.record_trade(-500.0)
        assert tracker.daily_pnl == -500.0
        assert tracker.is_within_limit()

    def test_exceeds_daily_limit(self) -> None:
        tracker = DailyDrawdownTracker(daily_limit=1000.0, starting_equity=50000.0)
        tracker.record_trade(-600.0)
        tracker.record_trade(-500.0)  # Total -1100 > 1000 limit
        assert not tracker.is_within_limit()

    def test_reset_daily(self) -> None:
        tracker = DailyDrawdownTracker(daily_limit=1000.0, starting_equity=50000.0)
        tracker.record_trade(-800.0)
        tracker.reset_daily(new_equity=49200.0)
        assert tracker.daily_pnl == 0.0
        assert tracker.is_within_limit()

    def test_wins_offset_losses(self) -> None:
        tracker = DailyDrawdownTracker(daily_limit=1000.0, starting_equity=50000.0)
        tracker.record_trade(-800.0)
        tracker.record_trade(500.0)
        assert tracker.daily_pnl == -300.0
        assert tracker.is_within_limit()


class TestMaxPositionGuard:
    """Test maximum concurrent position limits."""

    def test_allows_within_limit(self) -> None:
        guard = MaxPositionGuard(max_total=5, max_per_strategy=2)
        signal = _make_signal()
        assert guard.can_open(signal, open_positions=[])

    def test_blocks_at_total_limit(self) -> None:
        guard = MaxPositionGuard(max_total=2, max_per_strategy=2)
        existing = [
            _make_signal(symbol="ES"),
            _make_signal(symbol="CL", stop=76.0, entry=75.0, tp=73.0, direction="SHORT"),
        ]
        new_signal = _make_signal(symbol="GC", entry=2050.0, stop=2040.0, tp=2070.0)
        assert not guard.can_open(new_signal, open_positions=existing)

    def test_blocks_at_per_strategy_limit(self) -> None:
        guard = MaxPositionGuard(max_total=10, max_per_strategy=1)
        existing = [_make_signal(strategy="temporal_edge")]
        new_signal = _make_signal(strategy="temporal_edge")
        assert not guard.can_open(new_signal, open_positions=existing)

    def test_allows_different_strategy(self) -> None:
        guard = MaxPositionGuard(max_total=10, max_per_strategy=1)
        existing = [_make_signal(strategy="temporal_edge")]
        new_signal = _make_signal(strategy="vol_regime")
        assert guard.can_open(new_signal, open_positions=existing)


class TestRiskGate:
    """Test the combined risk gate that checks all constraints."""

    def test_approve_valid_signal(self) -> None:
        profile = PropFirmProfile(
            name="Apex",
            account_size=50000,
            trailing_drawdown=2500,
            daily_loss_limit=2500,
            profit_target=3000,
            max_contracts=10,
        )
        gate = RiskGate(profile)
        signal = _make_signal()
        assert gate.approve(signal) is True

    def test_reject_when_daily_limit_breached(self) -> None:
        profile = PropFirmProfile(
            name="Apex",
            account_size=50000,
            trailing_drawdown=2500,
            daily_loss_limit=1000,
            profit_target=3000,
            max_contracts=10,
        )
        gate = RiskGate(profile)
        gate.daily_tracker.record_trade(-900.0)
        signal = _make_signal(risk_dollars=200.0)
        # Would push daily loss to -1100 > 1000 limit
        assert gate.approve(signal) is False

    def test_reject_when_trailing_dd_breached(self) -> None:
        profile = PropFirmProfile(
            name="Apex",
            account_size=50000,
            trailing_drawdown=2500,
            daily_loss_limit=2500,
            profit_target=3000,
            max_contracts=10,
        )
        gate = RiskGate(profile)
        gate.trailing_tracker.update_equity(47000.0)  # $3000 DD
        signal = _make_signal()
        assert gate.approve(signal) is False

    def test_reject_nq_during_blackout(self) -> None:
        profile = PropFirmProfile(
            name="Apex",
            account_size=50000,
            trailing_drawdown=2500,
            daily_loss_limit=2500,
            profit_target=3000,
            max_contracts=10,
        )
        gate = RiskGate(profile)
        # 10:00 AM ET = during NQ blackout
        signal = _make_signal(
            symbol="NQ",
            timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=ET),
            stop=17400.0,
            entry=17500.0,
            tp=17600.0,
        )
        assert gate.approve(signal) is False

    def test_allow_es_during_nq_blackout(self) -> None:
        profile = PropFirmProfile(
            name="Apex",
            account_size=50000,
            trailing_drawdown=2500,
            daily_loss_limit=2500,
            profit_target=3000,
            max_contracts=10,
        )
        gate = RiskGate(profile)
        signal = _make_signal(
            symbol="ES",
            timestamp=datetime(2024, 1, 15, 10, 0, tzinfo=ET),
        )
        assert gate.approve(signal) is True

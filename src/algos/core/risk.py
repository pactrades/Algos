"""Risk management — drawdown tracking, position guards, and signal gating."""

from __future__ import annotations

from algos.core.config import PropFirmProfile
from algos.core.session import should_suppress_nq_signal
from algos.core.signal import Signal


class TrailingDrawdownTracker:
    """Tracks peak equity and current drawdown from that peak."""

    def __init__(self, max_drawdown: float, starting_equity: float) -> None:
        self._max_drawdown = max_drawdown
        self._peak_equity = starting_equity
        self._current_equity = starting_equity

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def current_drawdown(self) -> float:
        return self._peak_equity - self._current_equity

    def update_equity(self, equity: float) -> None:
        """Update current equity; raises peak if new high."""
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

    def is_within_limit(self) -> bool:
        """True if drawdown is at or under the max allowed."""
        return self.current_drawdown <= self._max_drawdown


class DailyDrawdownTracker:
    """Tracks intraday P&L against a daily loss limit."""

    def __init__(self, daily_limit: float, starting_equity: float) -> None:
        self._daily_limit = daily_limit
        self._starting_equity = starting_equity
        self._daily_pnl: float = 0.0

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    def record_trade(self, pnl: float) -> None:
        """Record a trade's P&L for the current day."""
        self._daily_pnl += pnl

    def reset_daily(self, new_equity: float) -> None:
        """Reset for a new trading day."""
        self._starting_equity = new_equity
        self._daily_pnl = 0.0

    def is_within_limit(self) -> bool:
        """True if daily loss is at or under the limit."""
        return abs(min(self._daily_pnl, 0.0)) <= self._daily_limit


class MaxPositionGuard:
    """Enforces maximum concurrent position limits."""

    def __init__(self, max_total: int, max_per_strategy: int) -> None:
        self._max_total = max_total
        self._max_per_strategy = max_per_strategy

    def can_open(self, signal: Signal, open_positions: list[Signal]) -> bool:
        """Check whether a new position can be opened."""
        if len(open_positions) >= self._max_total:
            return False
        strategy_count = sum(
            1 for p in open_positions if p.strategy == signal.strategy
        )
        return strategy_count < self._max_per_strategy


class RiskGate:
    """Combined risk gate that checks all constraints before approving a signal."""

    def __init__(self, profile: PropFirmProfile) -> None:
        self._profile = profile
        self.trailing_tracker = TrailingDrawdownTracker(
            max_drawdown=profile.trailing_drawdown,
            starting_equity=profile.account_size,
        )
        self.daily_tracker = DailyDrawdownTracker(
            daily_limit=profile.daily_loss_limit,
            starting_equity=profile.account_size,
        )

    def approve(self, signal: Signal) -> bool:
        """Approve or reject a signal based on all risk constraints.

        Checks:
        1. Trailing drawdown limit
        2. Daily loss limit (including projected loss from this signal)
        3. NQ blackout window
        """
        # Check trailing drawdown
        if not self.trailing_tracker.is_within_limit():
            return False

        # Check if this trade would push daily loss past the limit
        projected_pnl = self.daily_tracker.daily_pnl - signal.risk_dollars
        if abs(min(projected_pnl, 0.0)) > self._profile.daily_loss_limit:
            return False

        # Check NQ blackout
        return not should_suppress_nq_signal(signal.timestamp, signal.symbol)

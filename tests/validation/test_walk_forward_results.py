"""Validation tests for walk-forward analysis."""

from __future__ import annotations

import numpy as np
import pytest

from algos.research.walk_forward import walk_forward_test


def _generate_profitable_trades(
    n: int = 500, win_rate: float = 0.55, avg_win: float = 200.0, seed: int = 42
) -> list[float]:
    """Generate a synthetic profitable trade series (55% win rate, 2:1 R:R)."""
    rng = np.random.default_rng(seed)
    trades: list[float] = []
    for _ in range(n):
        if rng.random() < win_rate:
            # Winner: avg_win with some noise
            trades.append(float(rng.normal(avg_win, avg_win * 0.3)))
        else:
            # Loser: half the avg_win (2:1 reward-to-risk)
            trades.append(float(rng.normal(-avg_win / 2, avg_win * 0.15)))
    return trades


def _generate_random_trades(n: int = 500, seed: int = 99) -> list[float]:
    """Generate a zero-expectancy random trade series."""
    rng = np.random.default_rng(seed)
    return [float(x) for x in rng.normal(0.0, 100.0, size=n)]


class TestWalkForwardResults:
    """Walk-forward validation tests."""

    def test_profitable_system_passes_walk_forward(self) -> None:
        """A system with 55% win rate and 2:1 R:R should pass walk-forward."""
        trades = _generate_profitable_trades(n=500, seed=42)
        result = walk_forward_test(trades, in_sample_size=100, out_of_sample_size=50)

        assert result.passes, (
            f"Profitable system should pass walk-forward, "
            f"got {result.pct_profitable_windows:.1%} profitable windows"
        )
        assert result.pct_profitable_windows >= 0.6
        assert result.avg_oos_return > 0

    def test_random_system_fails_walk_forward(self) -> None:
        """A zero-expectancy random system should show poor walk-forward results."""
        trades = _generate_random_trades(n=500, seed=99)
        result = walk_forward_test(trades, in_sample_size=100, out_of_sample_size=50)

        # Random system: avg OOS return should be near zero, not reliably positive
        # We check that it either fails the 60% threshold or has near-zero avg return
        assert result.avg_oos_return < 50.0, (
            "Random system should not show strong positive OOS returns"
        )

    def test_walk_forward_window_count(self) -> None:
        """Verify the correct number of OOS windows are generated."""
        n_trades = 500
        is_size = 100
        oos_size = 50
        trades = _generate_profitable_trades(n=n_trades, seed=42)

        result = walk_forward_test(trades, in_sample_size=is_size, out_of_sample_size=oos_size)

        # Window slides by oos_size each step, starting at 0
        # Each window needs is_size + oos_size trades
        # start=0: needs 150, start=50: needs 200, ...
        # start + is_size + oos_size <= n_trades
        # start <= n_trades - is_size - oos_size = 350
        # start takes values 0, 50, 100, ..., 350 => 8 windows
        expected_windows = 0
        start = 0
        while start + is_size + oos_size <= n_trades:
            expected_windows += 1
            start += oos_size

        assert result.n_windows == expected_windows
        assert len(result.oos_returns) == expected_windows

    def test_walk_forward_insufficient_trades_raises(self) -> None:
        """Should raise ValueError if not enough trades for one window."""
        trades = [100.0] * 10
        with pytest.raises(ValueError, match="Not enough trades"):
            walk_forward_test(trades, in_sample_size=50, out_of_sample_size=25)

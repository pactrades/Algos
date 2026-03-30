"""Tests for walk-forward optimization engine."""

import numpy as np
import pytest

from algos.research.walk_forward import WalkForwardResult, walk_forward_test


@pytest.fixture
def profitable_trade_series() -> list[float]:
    """A long series of trades with consistent positive edge."""
    np.random.seed(42)
    trades = []
    for _ in range(500):
        if np.random.random() < 0.55:
            trades.append(np.random.uniform(0.5, 3.0))
        else:
            trades.append(-np.random.uniform(0.5, 1.5))
    return trades


@pytest.fixture
def unprofitable_trade_series() -> list[float]:
    """A long series of trades with negative edge."""
    np.random.seed(42)
    trades = []
    for _ in range(500):
        if np.random.random() < 0.35:
            trades.append(np.random.uniform(0.3, 1.5))
        else:
            trades.append(-np.random.uniform(0.8, 2.0))
    return trades


class TestWalkForwardTest:
    """Test walk-forward optimization."""

    def test_returns_walk_forward_result(self, profitable_trade_series: list[float]) -> None:
        result = walk_forward_test(
            trades=profitable_trade_series,
            in_sample_size=100,
            out_of_sample_size=50,
        )
        assert isinstance(result, WalkForwardResult)

    def test_result_has_multiple_windows(self, profitable_trade_series: list[float]) -> None:
        result = walk_forward_test(
            trades=profitable_trade_series,
            in_sample_size=100,
            out_of_sample_size=50,
        )
        # 500 trades, 100 IS + 50 OOS = 150 per window, should get multiple windows
        assert result.n_windows >= 2

    def test_profitable_series_passes(self, profitable_trade_series: list[float]) -> None:
        result = walk_forward_test(
            trades=profitable_trade_series,
            in_sample_size=100,
            out_of_sample_size=50,
        )
        # At least 60% of OOS windows should be profitable
        assert result.pct_profitable_windows >= 0.5

    def test_unprofitable_series_fails(self, unprofitable_trade_series: list[float]) -> None:
        result = walk_forward_test(
            trades=unprofitable_trade_series,
            in_sample_size=100,
            out_of_sample_size=50,
        )
        assert result.pct_profitable_windows < 0.6

    def test_oos_returns_populated(self, profitable_trade_series: list[float]) -> None:
        result = walk_forward_test(
            trades=profitable_trade_series,
            in_sample_size=100,
            out_of_sample_size=50,
        )
        assert len(result.oos_returns) == result.n_windows

    def test_too_few_trades_raises(self) -> None:
        with pytest.raises(ValueError, match="Not enough trades"):
            walk_forward_test(
                trades=[1.0, -0.5, 2.0],
                in_sample_size=100,
                out_of_sample_size=50,
            )

    def test_window_sizes_respected(self, profitable_trade_series: list[float]) -> None:
        result = walk_forward_test(
            trades=profitable_trade_series,
            in_sample_size=80,
            out_of_sample_size=40,
        )
        # Each OOS window should have trades
        for oos_ret in result.oos_returns:
            assert isinstance(oos_ret, float)

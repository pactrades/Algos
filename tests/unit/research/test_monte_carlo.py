"""Tests for Monte Carlo simulation."""

import numpy as np
import pytest

from algos.research.monte_carlo import MonteCarloResult, monte_carlo_simulate


@pytest.fixture
def profitable_trades() -> list[float]:
    """Trades with positive expectancy."""
    np.random.seed(42)
    trades = []
    for _ in range(200):
        if np.random.random() < 0.55:
            trades.append(np.random.uniform(0.5, 3.0))
        else:
            trades.append(-np.random.uniform(0.5, 1.5))
    return trades


@pytest.fixture
def risky_trades() -> list[float]:
    """Trades that are profitable but with high variance (risk of ruin)."""
    np.random.seed(42)
    trades = []
    for _ in range(200):
        if np.random.random() < 0.4:
            trades.append(np.random.uniform(2.0, 8.0))  # Big wins
        else:
            trades.append(-np.random.uniform(1.0, 4.0))  # Big losses
    return trades


class TestMonteCarloSimulate:
    """Test Monte Carlo trade sequence simulation."""

    def test_returns_monte_carlo_result(self, profitable_trades: list[float]) -> None:
        result = monte_carlo_simulate(
            trades=profitable_trades,
            n_simulations=1000,
            account_size=50000.0,
        )
        assert isinstance(result, MonteCarloResult)

    def test_result_has_required_fields(self, profitable_trades: list[float]) -> None:
        result = monte_carlo_simulate(
            trades=profitable_trades,
            n_simulations=1000,
            account_size=50000.0,
        )
        assert result.n_simulations == 1000
        assert result.median_final_equity > 0
        assert result.percentile_5_max_drawdown <= 0
        assert result.percentile_95_max_drawdown <= 0
        assert 0.0 <= result.probability_of_ruin <= 1.0

    def test_profitable_trades_grow_equity(self, profitable_trades: list[float]) -> None:
        result = monte_carlo_simulate(
            trades=profitable_trades,
            n_simulations=1000,
            account_size=50000.0,
        )
        assert result.median_final_equity > 50000.0

    def test_probability_of_ruin_low_for_good_system(self, profitable_trades: list[float]) -> None:
        result = monte_carlo_simulate(
            trades=profitable_trades,
            n_simulations=1000,
            account_size=50000.0,
            ruin_threshold=25000.0,  # 50% drawdown = ruin
        )
        assert result.probability_of_ruin < 0.1

    def test_max_drawdown_within_bounds(self, profitable_trades: list[float]) -> None:
        result = monte_carlo_simulate(
            trades=profitable_trades,
            n_simulations=1000,
            account_size=50000.0,
        )
        # 95th percentile DD should be negative (it's a drawdown)
        assert result.percentile_95_max_drawdown < 0
        # But shouldn't be worse than losing everything
        assert result.percentile_95_max_drawdown > -50000.0

    def test_n_simulations_respected(self, profitable_trades: list[float]) -> None:
        result = monte_carlo_simulate(
            trades=profitable_trades,
            n_simulations=500,
            account_size=50000.0,
        )
        assert result.n_simulations == 500

    def test_empty_trades_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            monte_carlo_simulate(
                trades=[],
                n_simulations=100,
                account_size=50000.0,
            )

    def test_median_return_is_float(self, profitable_trades: list[float]) -> None:
        result = monte_carlo_simulate(
            trades=profitable_trades,
            n_simulations=100,
            account_size=50000.0,
        )
        assert isinstance(result.median_annual_return, float)

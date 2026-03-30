"""Validation tests for Monte Carlo simulation results."""

from __future__ import annotations

import numpy as np

from algos.research.correlation import are_strategies_uncorrelated
from algos.research.monte_carlo import monte_carlo_simulate


def _generate_profitable_trades(
    n: int = 500, win_rate: float = 0.55, avg_win: float = 200.0, seed: int = 42
) -> list[float]:
    """Generate a synthetic profitable trade series (55% win rate, 2:1 R:R)."""
    rng = np.random.default_rng(seed)
    trades: list[float] = []
    for _ in range(n):
        if rng.random() < win_rate:
            trades.append(float(rng.normal(avg_win, avg_win * 0.3)))
        else:
            trades.append(float(rng.normal(-avg_win / 2, avg_win * 0.15)))
    return trades


class TestMonteCarloResults:
    """Monte Carlo simulation validation tests."""

    def test_profitable_system_monte_carlo(self) -> None:
        """A profitable system should show median final equity above starting equity."""
        trades = _generate_profitable_trades(n=500, seed=42)
        account_size = 50_000.0

        result = monte_carlo_simulate(trades, n_simulations=1000, account_size=account_size)

        assert result.n_simulations == 1000
        assert result.median_final_equity > account_size, (
            f"Median final equity ({result.median_final_equity:.0f}) "
            f"should exceed starting equity ({account_size:.0f})"
        )

    def test_monte_carlo_ruin_probability(self) -> None:
        """A good system with $50k account should have P(ruin) < 5%."""
        trades = _generate_profitable_trades(n=500, seed=42)
        account_size = 50_000.0

        result = monte_carlo_simulate(
            trades,
            n_simulations=1000,
            account_size=account_size,
            ruin_threshold=account_size * 0.5,
        )

        assert result.probability_of_ruin < 0.05, (
            f"P(ruin) should be < 5%, got {result.probability_of_ruin:.1%}"
        )

    def test_monte_carlo_drawdown_within_prop_firm_limits(self) -> None:
        """95th percentile max drawdown should stay within $2500 (Apex limit).

        We use a conservative system with small position sizes to ensure
        drawdowns remain within prop firm limits.
        """
        # Use smaller trade sizes to keep drawdowns within $2500
        trades = _generate_profitable_trades(n=500, win_rate=0.55, avg_win=50.0, seed=42)
        account_size = 50_000.0

        result = monte_carlo_simulate(trades, n_simulations=1000, account_size=account_size)

        # percentile_95_max_drawdown is negative (it's a drawdown)
        assert result.percentile_95_max_drawdown > -2500.0, (
            f"95th pct max DD ({result.percentile_95_max_drawdown:.0f}) "
            f"should be within -$2500 Apex limit"
        )
        assert result.passes_prop_firm

    def test_strategies_uncorrelated(self) -> None:
        """Five independent synthetic strategy return series should be uncorrelated."""
        rng = np.random.default_rng(seed=123)
        n_returns = 252

        strategy_returns: dict[str, list[float]] = {}
        strategy_names = [
            "momentum_breakout",
            "mean_reversion",
            "volatility_squeeze",
            "session_gap",
            "time_of_day",
        ]

        for name in strategy_names:
            # Each strategy has independent random returns
            returns = [float(x) for x in rng.normal(0.0, 1.0, size=n_returns)]
            strategy_returns[name] = returns

        assert are_strategies_uncorrelated(strategy_returns, threshold=0.3), (
            "Independent random strategy returns should be uncorrelated"
        )

"""Monte Carlo simulation — randomized trade sequence analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MonteCarloResult:
    """Results of a Monte Carlo simulation."""

    n_simulations: int
    median_final_equity: float
    median_annual_return: float
    percentile_5_max_drawdown: float  # 5th percentile (best case DD)
    percentile_95_max_drawdown: float  # 95th percentile (worst case DD)
    probability_of_ruin: float  # Fraction of sims that hit ruin threshold
    avg_max_drawdown: float

    @property
    def passes_prop_firm(self) -> bool:
        """Check if 95th percentile DD is within typical prop firm limits ($2500 on $50k)."""
        return self.percentile_95_max_drawdown > -2500.0


def monte_carlo_simulate(
    trades: list[float],
    n_simulations: int = 10000,
    account_size: float = 50000.0,
    ruin_threshold: float | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation by randomly shuffling trade order.

    For each simulation:
    1. Randomly shuffle the trade sequence
    2. Compute the equity curve
    3. Record max drawdown and final equity

    Args:
        trades: List of trade P&L values (in dollars).
        n_simulations: Number of random simulations to run.
        account_size: Starting account size in dollars.
        ruin_threshold: Equity level below which the account is "ruined".
            Defaults to 50% of account_size.

    Returns:
        MonteCarloResult with distribution statistics.

    Raises:
        ValueError: If trades list is empty.
    """
    if len(trades) == 0:
        raise ValueError("Cannot run Monte Carlo simulation with empty trade list.")

    if ruin_threshold is None:
        ruin_threshold = account_size * 0.5

    arr = np.asarray(trades, dtype=float)
    n_trades = len(arr)

    rng = np.random.default_rng(seed=42)

    final_equities = np.zeros(n_simulations)
    max_drawdowns = np.zeros(n_simulations)
    ruin_count = 0

    for i in range(n_simulations):
        # Shuffle trade order
        shuffled = rng.permutation(arr)

        # Build equity curve
        equity = account_size + np.cumsum(shuffled)

        # Track max drawdown
        running_peak = np.maximum.accumulate(np.concatenate(([account_size], equity)))
        drawdowns = np.concatenate(([account_size], equity)) - running_peak
        max_dd = float(np.min(drawdowns))

        final_equities[i] = equity[-1]
        max_drawdowns[i] = max_dd

        # Check ruin
        if np.any(equity < ruin_threshold):
            ruin_count += 1

    # Compute annual return (assuming ~252 trades per year as rough estimate)
    trades_per_year = min(252, n_trades)
    total_return_pct = final_equities / account_size - 1.0
    annual_factor = trades_per_year / n_trades if n_trades > 0 else 1.0
    annual_returns = total_return_pct * annual_factor

    return MonteCarloResult(
        n_simulations=n_simulations,
        median_final_equity=float(np.median(final_equities)),
        median_annual_return=float(np.median(annual_returns)),
        percentile_5_max_drawdown=float(np.percentile(max_drawdowns, 5)),
        percentile_95_max_drawdown=float(np.percentile(max_drawdowns, 95)),
        probability_of_ruin=ruin_count / n_simulations,
        avg_max_drawdown=float(np.mean(max_drawdowns)),
    )

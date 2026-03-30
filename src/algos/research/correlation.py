"""Cross-strategy correlation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pairwise_correlation(series_a: list[float], series_b: list[float]) -> float:
    """Compute Pearson correlation between two return series."""
    a = np.asarray(series_a, dtype=float)
    b = np.asarray(series_b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return 0.0

    # Use min length if different
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    corr_matrix = np.corrcoef(a, b)
    return float(corr_matrix[0, 1])


def strategy_correlation_matrix(
    strategy_returns: dict[str, list[float]],
) -> pd.DataFrame:
    """Compute the full correlation matrix across all strategies.

    Args:
        strategy_returns: Dict mapping strategy name -> list of returns.

    Returns:
        DataFrame with strategy names as both index and columns.
    """
    names = list(strategy_returns.keys())
    n = len(names)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0
            else:
                matrix[i, j] = pairwise_correlation(
                    strategy_returns[names[i]],
                    strategy_returns[names[j]],
                )

    return pd.DataFrame(matrix, index=names, columns=names)


def are_strategies_uncorrelated(
    strategy_returns: dict[str, list[float]],
    threshold: float = 0.3,
) -> bool:
    """Check if all pairwise strategy correlations are below a threshold.

    Args:
        strategy_returns: Dict mapping strategy name -> list of returns.
        threshold: Maximum allowed absolute correlation between any two strategies.

    Returns:
        True if all pairwise correlations are below the threshold.
    """
    matrix = strategy_correlation_matrix(strategy_returns)
    n = matrix.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix.iloc[i, j]) >= threshold:
                return False

    return True

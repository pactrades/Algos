"""Tests for cross-strategy correlation analysis."""

import numpy as np
import pytest

from algos.research.correlation import (
    are_strategies_uncorrelated,
    pairwise_correlation,
    strategy_correlation_matrix,
)


@pytest.fixture
def uncorrelated_strategies() -> dict[str, list[float]]:
    """Three strategy return series with low correlation."""
    np.random.seed(42)
    return {
        "temporal_edge": list(np.random.randn(200) * 0.01),
        "vol_regime": list(np.random.randn(200) * 0.015),
        "calendar_alpha": list(np.random.randn(200) * 0.008),
    }


@pytest.fixture
def correlated_strategies() -> dict[str, list[float]]:
    """Two strategy return series that are highly correlated."""
    np.random.seed(42)
    base = np.random.randn(200) * 0.01
    return {
        "strategy_a": list(base),
        "strategy_b": list(base + np.random.randn(200) * 0.001),  # Almost same
    }


class TestPairwiseCorrelation:
    """Test correlation between two return series."""

    def test_perfect_correlation(self) -> None:
        series = list(range(100))
        corr = pairwise_correlation(series, series)
        assert corr == pytest.approx(1.0, abs=0.01)

    def test_uncorrelated_series(self) -> None:
        np.random.seed(42)
        a = list(np.random.randn(1000))
        b = list(np.random.randn(1000))
        corr = pairwise_correlation(a, b)
        assert abs(corr) < 0.1

    def test_negative_correlation(self) -> None:
        a = list(range(100))
        b = list(range(100, 0, -1))
        corr = pairwise_correlation(a, b)
        assert corr < -0.9

    def test_returns_float(self) -> None:
        corr = pairwise_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert isinstance(corr, float)


class TestStrategyCorrelationMatrix:
    """Test full correlation matrix across strategies."""

    def test_matrix_is_square(self, uncorrelated_strategies: dict) -> None:
        matrix = strategy_correlation_matrix(uncorrelated_strategies)
        n = len(uncorrelated_strategies)
        assert matrix.shape == (n, n)

    def test_diagonal_is_one(self, uncorrelated_strategies: dict) -> None:
        matrix = strategy_correlation_matrix(uncorrelated_strategies)
        for i in range(matrix.shape[0]):
            assert matrix.iloc[i, i] == pytest.approx(1.0, abs=0.01)

    def test_symmetric(self, uncorrelated_strategies: dict) -> None:
        matrix = strategy_correlation_matrix(uncorrelated_strategies)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                assert matrix.iloc[i, j] == pytest.approx(matrix.iloc[j, i], abs=0.001)

    def test_uncorrelated_strategies_low_values(self, uncorrelated_strategies: dict) -> None:
        matrix = strategy_correlation_matrix(uncorrelated_strategies)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i != j:
                    assert abs(matrix.iloc[i, j]) < 0.3

    def test_correlated_strategies_high_values(self, correlated_strategies: dict) -> None:
        matrix = strategy_correlation_matrix(correlated_strategies)
        # Off-diagonal should be very high
        assert abs(matrix.iloc[0, 1]) > 0.9


class TestAreStrategiesUncorrelated:
    """Test the uncorrelation check."""

    def test_uncorrelated_passes(self, uncorrelated_strategies: dict) -> None:
        assert are_strategies_uncorrelated(uncorrelated_strategies, threshold=0.3) is True

    def test_correlated_fails(self, correlated_strategies: dict) -> None:
        assert are_strategies_uncorrelated(correlated_strategies, threshold=0.3) is False

    def test_custom_threshold(self, uncorrelated_strategies: dict) -> None:
        # Very strict threshold might fail even for random data
        result = are_strategies_uncorrelated(uncorrelated_strategies, threshold=0.01)
        assert isinstance(result, bool)

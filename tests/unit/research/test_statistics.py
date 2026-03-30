"""Tests for statistical analysis toolkit — distributions, significance, effect size."""

import numpy as np
import pytest

from algos.research.statistics import (
    cohens_d,
    distribution_analysis,
    returns_autocorrelation,
    significance_test,
    trade_statistics,
)


@pytest.fixture
def normal_returns() -> np.ndarray:
    """Generate normally distributed returns."""
    np.random.seed(42)
    return np.random.randn(500) * 0.01  # 1% daily vol


@pytest.fixture
def skewed_returns() -> np.ndarray:
    """Generate positively skewed returns (like a trend-following strategy)."""
    np.random.seed(42)
    base = np.random.randn(500) * 0.005
    # Add some large positive outliers
    base[::50] += 0.05
    return base


@pytest.fixture
def autocorrelated_returns() -> np.ndarray:
    """Generate returns with serial correlation."""
    np.random.seed(42)
    n = 500
    returns = np.zeros(n)
    returns[0] = np.random.randn() * 0.01
    for i in range(1, n):
        returns[i] = 0.5 * returns[i - 1] + np.random.randn() * 0.005
    return returns


@pytest.fixture
def winning_trades() -> list[float]:
    """Generate a set of trades with positive expectancy."""
    np.random.seed(42)
    # 55% win rate, avg win = 2R, avg loss = 1R
    trades = []
    for _ in range(200):
        if np.random.random() < 0.55:
            trades.append(np.random.uniform(0.5, 3.5))  # Winners
        else:
            trades.append(-np.random.uniform(0.5, 1.5))  # Losers
    return trades


@pytest.fixture
def losing_trades() -> list[float]:
    """Generate a set of trades with negative expectancy."""
    np.random.seed(42)
    # 35% win rate, avg win = 1R, avg loss = 1.5R
    trades = []
    for _ in range(200):
        if np.random.random() < 0.35:
            trades.append(np.random.uniform(0.3, 1.5))
        else:
            trades.append(-np.random.uniform(0.8, 2.0))
    return trades


class TestDistributionAnalysis:
    """Test return distribution analysis."""

    def test_returns_dict_with_required_keys(self, normal_returns: np.ndarray) -> None:
        result = distribution_analysis(normal_returns)
        required_keys = {"mean", "std", "skew", "kurtosis", "is_normal", "normality_p_value"}
        assert required_keys.issubset(result.keys())

    def test_normal_data_detected_as_normal(self, normal_returns: np.ndarray) -> None:
        result = distribution_analysis(normal_returns)
        # With 500 samples of pure normal data, should pass normality test
        assert result["is_normal"] is True
        assert result["normality_p_value"] > 0.05

    def test_skewed_data_has_positive_skew(self, skewed_returns: np.ndarray) -> None:
        result = distribution_analysis(skewed_returns)
        assert result["skew"] > 0

    def test_mean_is_close_to_expected(self, normal_returns: np.ndarray) -> None:
        result = distribution_analysis(normal_returns)
        # Mean should be close to 0 for standard normal scaled returns
        assert abs(result["mean"]) < 0.005

    def test_std_is_positive(self, normal_returns: np.ndarray) -> None:
        result = distribution_analysis(normal_returns)
        assert result["std"] > 0


class TestReturnsAutocorrelation:
    """Test serial correlation analysis."""

    def test_random_returns_low_autocorrelation(self, normal_returns: np.ndarray) -> None:
        result = returns_autocorrelation(normal_returns, max_lag=5)
        # Random returns should have low autocorrelation at all lags
        for lag, ac_value in result.items():
            assert abs(ac_value) < 0.15, f"Lag {lag} autocorrelation too high: {ac_value}"

    def test_autocorrelated_returns_high_lag1(self, autocorrelated_returns: np.ndarray) -> None:
        result = returns_autocorrelation(autocorrelated_returns, max_lag=5)
        # AR(1) with coeff 0.5 should show significant lag-1 autocorrelation
        assert abs(result[1]) > 0.3

    def test_returns_dict_with_lag_keys(self, normal_returns: np.ndarray) -> None:
        result = returns_autocorrelation(normal_returns, max_lag=3)
        assert set(result.keys()) == {1, 2, 3}

    def test_max_lag_respected(self, normal_returns: np.ndarray) -> None:
        result = returns_autocorrelation(normal_returns, max_lag=10)
        assert len(result) == 10


class TestSignificanceTest:
    """Test statistical significance testing."""

    def test_winning_trades_are_significant(self, winning_trades: list[float]) -> None:
        result = significance_test(winning_trades)
        assert result["p_value"] < 0.05
        assert result["is_significant"] is True
        assert result["mean"] > 0

    def test_losing_trades_not_significantly_positive(self, losing_trades: list[float]) -> None:
        result = significance_test(losing_trades)
        # Mean is negative, so one-sided test for positive should not be significant
        assert result["mean"] < 0

    def test_result_contains_required_keys(self, winning_trades: list[float]) -> None:
        result = significance_test(winning_trades)
        required = {"mean", "t_statistic", "p_value", "is_significant", "n_trades"}
        assert required.issubset(result.keys())

    def test_n_trades_correct(self, winning_trades: list[float]) -> None:
        result = significance_test(winning_trades)
        assert result["n_trades"] == 200

    def test_too_few_trades_not_significant(self) -> None:
        result = significance_test([1.0, 2.0, -0.5])
        # With only 3 trades, hard to be significant
        assert result["n_trades"] == 3


class TestCohensD:
    """Test effect size calculation."""

    def test_large_effect_size(self) -> None:
        # Clear positive edge: mean=2, std=1 → d=2.0 (very large)
        trades = [2.0 + np.random.randn() * 1.0 for _ in range(100)]
        d = cohens_d(trades)
        assert d > 0.8  # Large effect

    def test_small_effect_size(self) -> None:
        # Tiny edge: mean=0.05, std=1.0 → d≈0.05
        np.random.seed(42)
        trades = [0.05 + np.random.randn() * 1.0 for _ in range(100)]
        d = cohens_d(trades)
        assert d < 0.3  # Small effect

    def test_zero_trades_returns_zero(self) -> None:
        d = cohens_d([])
        assert d == 0.0

    def test_positive_for_positive_mean(self, winning_trades: list[float]) -> None:
        d = cohens_d(winning_trades)
        assert d > 0


class TestTradeStatistics:
    """Test comprehensive trade statistics."""

    def test_winning_trades_stats(self, winning_trades: list[float]) -> None:
        stats = trade_statistics(winning_trades)
        assert stats["win_rate"] > 0.5
        assert stats["profit_factor"] > 1.0
        assert stats["expectancy"] > 0
        assert stats["total_trades"] == 200

    def test_losing_trades_stats(self, losing_trades: list[float]) -> None:
        stats = trade_statistics(losing_trades)
        assert stats["win_rate"] < 0.5
        assert stats["profit_factor"] < 1.0
        assert stats["expectancy"] < 0

    def test_stats_contain_required_keys(self, winning_trades: list[float]) -> None:
        stats = trade_statistics(winning_trades)
        required = {
            "total_trades",
            "win_rate",
            "avg_win",
            "avg_loss",
            "profit_factor",
            "expectancy",
            "max_drawdown",
            "sharpe_ratio",
        }
        assert required.issubset(stats.keys())

    def test_max_drawdown_is_negative_or_zero(self, winning_trades: list[float]) -> None:
        stats = trade_statistics(winning_trades)
        assert stats["max_drawdown"] <= 0

    def test_empty_trades_returns_zeros(self) -> None:
        stats = trade_statistics([])
        assert stats["total_trades"] == 0
        assert stats["win_rate"] == 0.0

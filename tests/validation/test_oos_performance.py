"""Validation tests for out-of-sample performance analysis."""

from __future__ import annotations

import numpy as np

from algos.research.statistics import significance_test, trade_statistics


def _generate_profitable_trades(
    n: int = 1000, win_rate: float = 0.55, avg_win: float = 200.0, seed: int = 42
) -> list[float]:
    """Generate a synthetic profitable trade series."""
    rng = np.random.default_rng(seed)
    trades: list[float] = []
    for _ in range(n):
        if rng.random() < win_rate:
            trades.append(float(rng.normal(avg_win, avg_win * 0.3)))
        else:
            trades.append(float(rng.normal(-avg_win / 2, avg_win * 0.15)))
    return trades


def _split_oos(trades: list[float], in_sample_frac: float = 0.7) -> tuple[list[float], list[float]]:
    """Split trades into in-sample and out-of-sample portions."""
    split_idx = int(len(trades) * in_sample_frac)
    return trades[:split_idx], trades[split_idx:]


class TestOOSPerformance:
    """Out-of-sample performance validation tests."""

    def test_oos_split_preserves_data(self) -> None:
        """A 70/30 split of 1000 trades should produce correct sizes."""
        trades = _generate_profitable_trades(n=1000, seed=42)
        in_sample, oos = _split_oos(trades, in_sample_frac=0.7)

        assert len(in_sample) == 700
        assert len(oos) == 300
        assert len(in_sample) + len(oos) == len(trades)
        # Verify data integrity — concatenation should match original
        assert in_sample + oos == trades

    def test_profitable_system_oos_positive(self) -> None:
        """A profitable system should have positive returns in the OOS portion."""
        trades = _generate_profitable_trades(n=1000, seed=42)
        _, oos = _split_oos(trades, in_sample_frac=0.7)

        oos_total = sum(oos)
        assert oos_total > 0, (
            f"OOS total return should be positive for a profitable system, got {oos_total:.2f}"
        )

        oos_mean = np.mean(oos)
        assert oos_mean > 0, "OOS mean return should be positive"

    def test_statistics_on_oos_sample(self) -> None:
        """trade_statistics() on OOS data should return all expected metrics."""
        trades = _generate_profitable_trades(n=1000, seed=42)
        _, oos = _split_oos(trades, in_sample_frac=0.7)

        stats = trade_statistics(oos)

        expected_keys = {
            "total_trades",
            "win_rate",
            "avg_win",
            "avg_loss",
            "profit_factor",
            "expectancy",
            "max_drawdown",
            "sharpe_ratio",
        }
        assert set(stats.keys()) == expected_keys

        assert stats["total_trades"] == 300
        assert 0.0 <= stats["win_rate"] <= 1.0
        assert stats["avg_win"] > 0
        assert stats["avg_loss"] <= 0
        assert stats["profit_factor"] > 0
        assert stats["expectancy"] > 0
        assert stats["max_drawdown"] <= 0

    def test_significance_on_oos(self) -> None:
        """significance_test() on OOS trades should return complete result structure."""
        trades = _generate_profitable_trades(n=1000, seed=42)
        _, oos = _split_oos(trades, in_sample_frac=0.7)

        result = significance_test(oos)

        expected_keys = {"mean", "t_statistic", "p_value", "is_significant", "n_trades"}
        assert set(result.keys()) == expected_keys

        assert result["n_trades"] == 300
        assert isinstance(result["t_statistic"], float)
        assert isinstance(result["p_value"], float)
        assert 0.0 <= result["p_value"] <= 1.0

        # A profitable system with 300 OOS trades should be significant
        assert result["is_significant"], (
            f"Profitable system OOS should be significant, p={result['p_value']:.4f}"
        )
        assert result["mean"] > 0

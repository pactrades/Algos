"""Statistical analysis toolkit — distributions, significance tests, effect size."""

from __future__ import annotations

import numpy as np
from scipy import stats


def distribution_analysis(returns: np.ndarray) -> dict[str, object]:
    """Analyze the distribution of a return series.

    Returns dict with: mean, std, skew, kurtosis, is_normal, normality_p_value.
    """
    returns = np.asarray(returns, dtype=float)

    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    skew = float(stats.skew(returns)) if len(returns) > 2 else 0.0
    kurtosis = float(stats.kurtosis(returns)) if len(returns) > 3 else 0.0

    # Shapiro-Wilk normality test (works well for n < 5000)
    if len(returns) >= 8:
        _, p_value = stats.shapiro(returns[:5000])  # Cap at 5000 for performance
        is_normal = bool(p_value > 0.05)
    else:
        p_value = 1.0
        is_normal = True

    return {
        "mean": mean,
        "std": std,
        "skew": skew,
        "kurtosis": kurtosis,
        "is_normal": is_normal,
        "normality_p_value": float(p_value),
    }


def returns_autocorrelation(returns: np.ndarray, max_lag: int = 5) -> dict[int, float]:
    """Compute autocorrelation of returns at multiple lags.

    Returns dict mapping lag -> autocorrelation value.
    """
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    mean = np.mean(returns)
    var = np.var(returns)

    result: dict[int, float] = {}
    for lag in range(1, max_lag + 1):
        if lag >= n or var == 0:
            result[lag] = 0.0
        else:
            covar = np.mean((returns[lag:] - mean) * (returns[:-lag] - mean))
            result[lag] = float(covar / var)

    return result


def significance_test(trades: list[float]) -> dict[str, object]:
    """Test if trade returns are statistically significantly different from zero.

    Uses a one-sample t-test (H0: mean = 0, H1: mean > 0).

    Returns dict with: mean, t_statistic, p_value, is_significant, n_trades.
    """
    n = len(trades)
    if n == 0:
        return {
            "mean": 0.0,
            "t_statistic": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "n_trades": 0,
        }

    arr = np.asarray(trades, dtype=float)
    mean = float(np.mean(arr))

    if n < 2 or np.std(arr, ddof=1) == 0:
        return {
            "mean": mean,
            "t_statistic": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "n_trades": n,
        }

    t_stat, p_two_sided = stats.ttest_1samp(arr, 0.0)
    # One-sided p-value for H1: mean > 0
    p_one_sided = p_two_sided / 2 if t_stat > 0 else 1.0 - p_two_sided / 2

    return {
        "mean": mean,
        "t_statistic": float(t_stat),
        "p_value": float(p_one_sided),
        "is_significant": bool(p_one_sided < 0.05),
        "n_trades": n,
    }


def cohens_d(trades: list[float]) -> float:
    """Calculate Cohen's d effect size for trade returns vs zero.

    d = mean / std. Interpretation: 0.2=small, 0.5=medium, 0.8=large.
    """
    if len(trades) == 0:
        return 0.0

    arr = np.asarray(trades, dtype=float)
    std = float(np.std(arr, ddof=1))
    if std == 0:
        return 0.0

    return float(np.mean(arr) / std)


def trade_statistics(trades: list[float]) -> dict[str, float]:
    """Compute comprehensive trade statistics.

    Returns dict with: total_trades, win_rate, avg_win, avg_loss,
    profit_factor, expectancy, max_drawdown, sharpe_ratio.
    """
    if len(trades) == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
        }

    arr = np.asarray(trades, dtype=float)
    winners = arr[arr > 0]
    losers = arr[arr <= 0]

    total = len(arr)
    win_rate = len(winners) / total if total > 0 else 0.0
    avg_win = float(np.mean(winners)) if len(winners) > 0 else 0.0
    avg_loss = float(np.mean(losers)) if len(losers) > 0 else 0.0

    gross_profit = float(np.sum(winners)) if len(winners) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losers))) if len(losers) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    expectancy = float(np.mean(arr))

    # Max drawdown from cumulative equity curve
    cumulative = np.cumsum(arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_drawdown = float(np.min(drawdowns))

    # Sharpe ratio (annualized assuming ~252 trading days)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    sharpe_ratio = (expectancy / std) * np.sqrt(252) if std > 0 else 0.0

    return {
        "total_trades": total,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": float(sharpe_ratio),
    }

"""Walk-forward optimization engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WalkForwardResult:
    """Results of a walk-forward test."""

    n_windows: int
    oos_returns: list[float]  # Sum of returns in each OOS window
    pct_profitable_windows: float  # Fraction of OOS windows with positive return
    avg_oos_return: float
    worst_oos_return: float
    best_oos_return: float

    @property
    def passes(self) -> bool:
        """Check if walk-forward passes the 60% profitable windows threshold."""
        return self.pct_profitable_windows >= 0.6


def walk_forward_test(
    trades: list[float],
    in_sample_size: int,
    out_of_sample_size: int,
) -> WalkForwardResult:
    """Run a walk-forward test on a trade series.

    Slides a rolling window through the trade series:
    - in_sample_size trades for "fitting" (we just measure performance)
    - out_of_sample_size trades for "validation"

    The window advances by out_of_sample_size each step.

    Args:
        trades: List of trade P&L values.
        in_sample_size: Number of trades in each in-sample window.
        out_of_sample_size: Number of trades in each out-of-sample window.

    Returns:
        WalkForwardResult with per-window OOS performance.

    Raises:
        ValueError: If not enough trades for at least one full window.
    """
    total = len(trades)
    window_size = in_sample_size + out_of_sample_size

    if total < window_size:
        raise ValueError(
            f"Not enough trades ({total}) for walk-forward test. "
            f"Need at least {window_size} (IS={in_sample_size} + OOS={out_of_sample_size})."
        )

    arr = np.asarray(trades, dtype=float)
    oos_returns: list[float] = []

    start = 0
    while start + window_size <= total:
        oos_start = start + in_sample_size
        oos_end = oos_start + out_of_sample_size
        oos_return = float(np.sum(arr[oos_start:oos_end]))
        oos_returns.append(oos_return)
        start += out_of_sample_size  # Slide forward by OOS size

    n_windows = len(oos_returns)
    profitable = sum(1 for r in oos_returns if r > 0)

    return WalkForwardResult(
        n_windows=n_windows,
        oos_returns=oos_returns,
        pct_profitable_windows=profitable / n_windows if n_windows > 0 else 0.0,
        avg_oos_return=float(np.mean(oos_returns)) if oos_returns else 0.0,
        worst_oos_return=float(np.min(oos_returns)) if oos_returns else 0.0,
        best_oos_return=float(np.max(oos_returns)) if oos_returns else 0.0,
    )

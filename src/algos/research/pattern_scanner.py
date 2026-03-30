"""Pattern scanner framework for discovering statistical edges in market data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PatternCandidate:
    """A candidate pattern discovered by a scanner."""

    name: str
    description: str
    n_occurrences: int
    avg_return: float
    win_rate: float
    p_value: float
    effect_size: float

    def passes_minimum_criteria(self) -> bool:
        """Check whether this candidate meets minimum thresholds for further validation."""
        if self.n_occurrences < 100:
            return False
        if self.p_value >= 0.05:
            return False
        return self.effect_size >= 0.2


class BasePatternScanner(ABC):
    """Abstract base class for all pattern scanners."""

    @abstractmethod
    def scan(self, data: pd.DataFrame) -> list[PatternCandidate]:
        """Scan *data* for patterns and return a list of candidates.

        Parameters
        ----------
        data:
            DataFrame with columns ``open``, ``high``, ``low``, ``close``,
            ``volume`` and a :class:`~pandas.DatetimeIndex` named ``timestamp``.
        """
        ...


class TimeOfDayScanner(BasePatternScanner):
    """Scan intraday data for time-of-day return patterns."""

    def scan(self, data: pd.DataFrame) -> list[PatternCandidate]:
        df = data.copy()
        df["return"] = df["close"].pct_change()
        df = df.dropna(subset=["return"])

        # Bucket by hour
        df["hour"] = df.index.hour  # type: ignore[union-attr]

        candidates: list[PatternCandidate] = []
        overall_mean = df["return"].mean()

        for hour, group in df.groupby("hour"):
            returns = group["return"].values
            n = len(returns)
            if n < 2:
                continue

            avg_ret = float(np.mean(returns))
            win_rate = float(np.mean(returns > 0))

            # One-sample t-test against overall mean
            t_stat, p_val = stats.ttest_1samp(returns, overall_mean)
            p_val = float(p_val)

            std = float(np.std(returns, ddof=1))
            effect = abs(avg_ret - overall_mean) / std if std > 0 else 0.0

            candidates.append(
                PatternCandidate(
                    name=f"tod_hour_{hour}",
                    description=f"Return pattern at hour {hour}",
                    n_occurrences=n,
                    avg_return=avg_ret,
                    win_rate=win_rate,
                    p_value=p_val,
                    effect_size=effect,
                )
            )

        return candidates


class VolatilitySqueezeScanner(BasePatternScanner):
    """Scan for volatility compression/expansion patterns using ATR."""

    def __init__(self, atr_period: int = 14, squeeze_threshold: float = 0.75) -> None:
        self.atr_period = atr_period
        self.squeeze_threshold = squeeze_threshold

    def scan(self, data: pd.DataFrame) -> list[PatternCandidate]:
        df = data.copy()

        # Compute ATR
        high = df["high"]
        low = df["low"]
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        # Rolling ATR ratio: current ATR vs longer-window ATR
        atr_long = atr.rolling(self.atr_period * 4).mean()
        ratio = atr / atr_long
        df["atr_ratio"] = ratio

        # Forward 1-bar return
        df["fwd_return"] = df["close"].shift(-1) / df["close"] - 1
        df = df.dropna(subset=["atr_ratio", "fwd_return"])

        candidates: list[PatternCandidate] = []

        # Squeeze: low volatility relative to recent history
        squeeze_mask = df["atr_ratio"] <= self.squeeze_threshold
        squeeze_returns = df.loc[squeeze_mask, "fwd_return"].values
        non_squeeze_returns = df.loc[~squeeze_mask, "fwd_return"].values

        if len(squeeze_returns) >= 2 and len(non_squeeze_returns) >= 2:
            avg_ret = float(np.mean(np.abs(squeeze_returns)))
            win_rate = float(np.mean(squeeze_returns > 0))
            t_stat, p_val = stats.ttest_ind(
                np.abs(squeeze_returns), np.abs(non_squeeze_returns), equal_var=False
            )
            combined = np.concatenate([squeeze_returns, non_squeeze_returns])
            pooled_std = float(np.std(combined, ddof=1))
            effect = (
                abs(float(np.mean(squeeze_returns)) - float(np.mean(non_squeeze_returns)))
                / pooled_std
                if pooled_std > 0
                else 0.0
            )

            candidates.append(
                PatternCandidate(
                    name="volatility_squeeze",
                    description=(
                        f"ATR ratio <= {self.squeeze_threshold} "
                        f"(period={self.atr_period}) predicts expansion"
                    ),
                    n_occurrences=int(squeeze_mask.sum()),
                    avg_return=avg_ret,
                    win_rate=win_rate,
                    p_value=float(p_val),
                    effect_size=effect,
                )
            )

        # Expansion: high volatility
        expansion_threshold = 2.0 - self.squeeze_threshold
        expansion_mask = df["atr_ratio"] >= expansion_threshold
        expansion_returns = df.loc[expansion_mask, "fwd_return"].values
        rest_returns = df.loc[~expansion_mask, "fwd_return"].values

        if len(expansion_returns) >= 2 and len(rest_returns) >= 2:
            avg_ret = float(np.mean(expansion_returns))
            win_rate = float(np.mean(expansion_returns > 0))
            t_stat, p_val = stats.ttest_ind(expansion_returns, rest_returns, equal_var=False)
            pooled_std = float(
                np.std(np.concatenate([expansion_returns, rest_returns]), ddof=1)
            )
            effect = (
                abs(float(np.mean(expansion_returns)) - float(np.mean(rest_returns)))
                / pooled_std
                if pooled_std > 0
                else 0.0
            )

            candidates.append(
                PatternCandidate(
                    name="volatility_expansion",
                    description=(
                        f"ATR ratio >= {expansion_threshold} "
                        f"(period={self.atr_period}) mean-reversion"
                    ),
                    n_occurrences=int(expansion_mask.sum()),
                    avg_return=avg_ret,
                    win_rate=win_rate,
                    p_value=float(p_val),
                    effect_size=effect,
                )
            )

        return candidates


class SessionGapScanner(BasePatternScanner):
    """Scan for session gap (open vs previous close) patterns."""

    def scan(self, data: pd.DataFrame) -> list[PatternCandidate]:
        df = data.copy()

        # Gap: today's open vs yesterday's close
        prev_close = df["close"].shift(1)
        df["gap_pct"] = (df["open"] - prev_close) / prev_close
        # Intraday return from open to close
        df["intraday_return"] = (df["close"] - df["open"]) / df["open"]
        df = df.dropna(subset=["gap_pct", "intraday_return"])

        candidates: list[PatternCandidate] = []

        # Gap-up: does a gap up lead to continuation or fade?
        gap_up_mask = df["gap_pct"] > 0
        gap_up_returns = df.loc[gap_up_mask, "intraday_return"].values
        gap_down_returns = df.loc[~gap_up_mask, "intraday_return"].values

        if len(gap_up_returns) >= 2:
            avg_ret = float(np.mean(gap_up_returns))
            win_rate = float(np.mean(gap_up_returns > 0))
            t_stat, p_val = stats.ttest_1samp(gap_up_returns, 0.0)
            std = float(np.std(gap_up_returns, ddof=1))
            effect = abs(avg_ret) / std if std > 0 else 0.0

            candidates.append(
                PatternCandidate(
                    name="gap_up_intraday",
                    description="Intraday return following a gap-up open",
                    n_occurrences=int(gap_up_mask.sum()),
                    avg_return=avg_ret,
                    win_rate=win_rate,
                    p_value=float(p_val),
                    effect_size=effect,
                )
            )

        # Gap-down
        if len(gap_down_returns) >= 2:
            avg_ret = float(np.mean(gap_down_returns))
            win_rate = float(np.mean(gap_down_returns > 0))
            t_stat, p_val = stats.ttest_1samp(gap_down_returns, 0.0)
            std = float(np.std(gap_down_returns, ddof=1))
            effect = abs(avg_ret) / std if std > 0 else 0.0

            candidates.append(
                PatternCandidate(
                    name="gap_down_intraday",
                    description="Intraday return following a gap-down open",
                    n_occurrences=int((~gap_up_mask).sum()),
                    avg_return=avg_ret,
                    win_rate=win_rate,
                    p_value=float(p_val),
                    effect_size=effect,
                )
            )

        # Large gap (top quartile by absolute gap size)
        abs_gap = df["gap_pct"].abs()
        q75 = abs_gap.quantile(0.75)
        large_gap_mask = abs_gap >= q75
        large_gap_returns = df.loc[large_gap_mask, "intraday_return"].values
        small_gap_returns = df.loc[~large_gap_mask, "intraday_return"].values

        if len(large_gap_returns) >= 2 and len(small_gap_returns) >= 2:
            avg_ret = float(np.mean(large_gap_returns))
            win_rate = float(np.mean(large_gap_returns > 0))
            t_stat, p_val = stats.ttest_ind(
                large_gap_returns, small_gap_returns, equal_var=False
            )
            pooled_std = float(
                np.std(np.concatenate([large_gap_returns, small_gap_returns]), ddof=1)
            )
            effect = (
                abs(float(np.mean(large_gap_returns)) - float(np.mean(small_gap_returns)))
                / pooled_std
                if pooled_std > 0
                else 0.0
            )

            candidates.append(
                PatternCandidate(
                    name="large_gap_fade",
                    description="Intraday return following a large session gap (top quartile)",
                    n_occurrences=int(large_gap_mask.sum()),
                    avg_return=avg_ret,
                    win_rate=win_rate,
                    p_value=float(p_val),
                    effect_size=effect,
                )
            )

        return candidates

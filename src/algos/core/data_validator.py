"""Data validation — gap detection, outlier filtering, OHLCV integrity checks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    """Results of OHLCV data validation."""

    is_valid: bool
    row_count: int
    issues: dict[str, list[int]] = field(default_factory=dict)

    def add_issue(self, issue_type: str, indices: list[int]) -> None:
        """Record an issue with the affected row indices."""
        if indices:
            self.issues[issue_type] = indices
            self.is_valid = False


def detect_gaps(
    df: pd.DataFrame, expected_freq: str = "5min"
) -> list[dict[str, object]]:
    """Detect gaps (missing bars) in a time-indexed OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        expected_freq: Expected bar frequency (e.g., "5min", "1min", "1h").

    Returns:
        List of gap dicts with keys: "start", "end", "missing_bars".
    """
    if len(df) < 2:
        return []

    expected_delta = pd.Timedelta(expected_freq)
    time_diffs = df.index.to_series().diff()

    gaps: list[dict[str, object]] = []
    for i in range(1, len(time_diffs)):
        diff = time_diffs.iloc[i]
        if diff > expected_delta:
            missing = int(diff / expected_delta) - 1
            if missing > 0:
                gaps.append(
                    {
                        "start": df.index[i - 1],
                        "end": df.index[i],
                        "missing_bars": missing,
                    }
                )

    return gaps


def detect_outliers(
    df: pd.DataFrame, z_threshold: float = 4.0
) -> list[int]:
    """Detect price outliers using z-score on close returns.

    Args:
        df: OHLCV DataFrame with a "close" column.
        z_threshold: Z-score threshold for outlier detection.

    Returns:
        List of integer positional indices where outliers were detected.
    """
    if len(df) < 3:
        return []

    returns = df["close"].pct_change().dropna()
    if returns.std() == 0:
        return []

    z_scores = np.abs((returns - returns.mean()) / returns.std())
    outlier_mask = z_scores > z_threshold

    # Map back to positional indices in the original DataFrame
    # returns starts at index 1 (first diff is NaN), so offset by 1
    outlier_positions = [i + 1 for i, is_outlier in enumerate(outlier_mask) if is_outlier]
    return outlier_positions


def validate_ohlcv_integrity(df: pd.DataFrame) -> ValidationReport:
    """Run integrity checks on an OHLCV DataFrame.

    Checks:
    - No NaN values in OHLCV columns
    - High >= Low for every bar
    - Open and Close within [Low, High] range
    - Volume >= 0

    Returns:
        ValidationReport with is_valid flag and issue details.
    """
    report = ValidationReport(is_valid=True, row_count=len(df))

    # Check for NaN values
    nan_mask = df[["open", "high", "low", "close", "volume"]].isna().any(axis=1)
    nan_indices = list(np.where(nan_mask)[0])
    report.add_issue("nan_values", nan_indices)

    # Check high >= low
    high_below_low = list(np.where(df["high"] < df["low"])[0])
    report.add_issue("high_below_low", high_below_low)

    # Check open within [low, high]
    open_outside = list(
        np.where((df["open"] > df["high"]) | (df["open"] < df["low"]))[0]
    )
    report.add_issue("open_outside_range", open_outside)

    # Check close within [low, high]
    close_outside = list(
        np.where((df["close"] > df["high"]) | (df["close"] < df["low"]))[0]
    )
    report.add_issue("close_outside_range", close_outside)

    # Check volume >= 0
    neg_volume = list(np.where(df["volume"] < 0)[0])
    report.add_issue("negative_volume", neg_volume)

    return report

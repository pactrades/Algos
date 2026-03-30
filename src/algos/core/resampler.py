"""Timeframe resampling — convert OHLCV data between timeframes."""

from __future__ import annotations

import pandas as pd

# Valid timeframes and their pandas offset aliases
VALID_TIMEFRAMES: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}

# OHLCV aggregation rules
_AGG_RULES = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to a target timeframe.

    Args:
        df: OHLCV DataFrame with DatetimeIndex named "timestamp".
            Expected columns: ["open", "high", "low", "close", "volume"]
        timeframe: Target timeframe, one of VALID_TIMEFRAMES keys.

    Returns:
        Resampled OHLCV DataFrame with same column structure.

    Raises:
        ValueError: If timeframe is not in VALID_TIMEFRAMES.
    """
    if timeframe not in VALID_TIMEFRAMES:
        raise ValueError(
            f"Invalid timeframe: '{timeframe}'. "
            f"Must be one of: {', '.join(sorted(VALID_TIMEFRAMES.keys()))}"
        )

    offset = VALID_TIMEFRAMES[timeframe]

    # If same timeframe as input frequency, return a copy
    if df.index.freq is not None and df.index.freqstr == offset:
        return df.copy()

    # For "1m" on 1-min data, detect by checking the mode of time deltas
    if timeframe == "1m" and len(df) > 1:
        median_delta = df.index.to_series().diff().median()
        if median_delta is not None and median_delta <= pd.Timedelta(minutes=1):
            return df.copy()

    resampled = (
        df.resample(offset, label="left", closed="left")
        .agg(_AGG_RULES)
        .dropna(how="all")
    )

    resampled.index.name = "timestamp"
    return resampled

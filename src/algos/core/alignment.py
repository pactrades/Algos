"""Multi-instrument data alignment — synchronize timestamps across instruments."""

from __future__ import annotations

import pandas as pd


def find_common_timerange(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Find the common time range across all instruments.

    Returns the (start, end) timestamps that are present in ALL instruments.
    Start = max of all earliest timestamps.
    End = min of all latest timestamps.

    Raises ValueError if data is empty.
    """
    if not data:
        raise ValueError("Cannot find common timerange: data dict is empty")

    starts = [df.index.min() for df in data.values()]
    ends = [df.index.max() for df in data.values()]

    return max(starts), min(ends)


def align_instruments(
    data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Align multiple instruments to share the same timestamps (inner join).

    Only timestamps present in ALL instruments are kept.
    Each DataFrame is expected to have a DatetimeIndex.

    Args:
        data: Dict mapping symbol -> OHLCV DataFrame with DatetimeIndex.

    Returns:
        Dict mapping symbol -> aligned OHLCV DataFrame.
    """
    if not data:
        return {}

    if len(data) == 1:
        symbol = next(iter(data))
        return {symbol: data[symbol].copy()}

    # Find the intersection of all timestamps
    common_index: pd.DatetimeIndex | None = None
    for df in data.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)

    assert common_index is not None

    # Reindex each instrument to the common timestamps
    result: dict[str, pd.DataFrame] = {}
    for symbol, df in data.items():
        result[symbol] = df.loc[common_index].copy()

    return result

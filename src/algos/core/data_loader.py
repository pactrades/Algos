"""Data loader — parquet file loading, schema auto-detection, OHLCV normalization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Known column name mappings for auto-detection
_TIMESTAMP_ALIASES = {"timestamp", "time", "date", "datetime", "ts", "dt"}
_OPEN_ALIASES = {"open", "o"}
_HIGH_ALIASES = {"high", "h"}
_LOW_ALIASES = {"low", "l"}
_CLOSE_ALIASES = {"close", "c"}
_VOLUME_ALIASES = {"volume", "vol", "v"}


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file and return as a DataFrame.

    Raises FileNotFoundError if path doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    return pd.read_parquet(path)


def _match_column(columns: list[str], aliases: set[str]) -> str | None:
    """Find the first column name that matches any alias (case-insensitive)."""
    col_lower_map = {col.lower(): col for col in columns}
    for alias in aliases:
        if alias in col_lower_map:
            return col_lower_map[alias]
    return None


def detect_schema(df: pd.DataFrame) -> dict[str, str]:
    """Auto-detect OHLCV column mapping from a DataFrame.

    Returns a dict mapping standard names to actual column names:
    {"timestamp": "Timestamp", "open": "Open", "high": "High", ...}

    Raises ValueError if required columns cannot be detected.
    """
    columns = list(df.columns)

    # Check if the index is already a datetime (timestamp might be the index)
    timestamp_col = _match_column(columns, _TIMESTAMP_ALIASES)

    mapping: dict[str, str | None] = {
        "timestamp": timestamp_col,
        "open": _match_column(columns, _OPEN_ALIASES),
        "high": _match_column(columns, _HIGH_ALIASES),
        "low": _match_column(columns, _LOW_ALIASES),
        "close": _match_column(columns, _CLOSE_ALIASES),
        "volume": _match_column(columns, _VOLUME_ALIASES),
    }

    # Check for required OHLC columns
    missing = [k for k in ["open", "high", "low", "close"] if mapping[k] is None]
    if missing:
        raise ValueError(
            f"Could not detect required OHLCV columns: {missing}. "
            f"Available columns: {columns}"
        )

    # Timestamp might be in the index
    if mapping["timestamp"] is None and isinstance(df.index, pd.DatetimeIndex):
        mapping["timestamp"] = df.index.name or "index"

    if mapping["timestamp"] is None:
        raise ValueError(
            f"Could not detect timestamp column. Available columns: {columns}"
        )

    return {k: v for k, v in mapping.items() if v is not None}


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame to standard OHLCV format.

    Output format:
    - Index: DatetimeIndex named "timestamp"
    - Columns: ["open", "high", "low", "close", "volume"]
    - Sorted ascending by timestamp
    - OHLC as float64
    """
    schema = detect_schema(df)

    # Build the normalized DataFrame
    result = pd.DataFrame()

    # Set the timestamp as index
    ts_col = schema["timestamp"]
    if ts_col in df.columns:
        result.index = pd.DatetimeIndex(pd.to_datetime(df[ts_col]), name="timestamp")
    elif isinstance(df.index, pd.DatetimeIndex):
        result.index = df.index
        result.index.name = "timestamp"
    else:
        result.index = pd.DatetimeIndex(pd.to_datetime(df.index), name="timestamp")

    # Map OHLCV columns
    for std_name in ["open", "high", "low", "close"]:
        result[std_name] = df[schema[std_name]].astype(float).values

    if "volume" in schema:
        result["volume"] = df[schema["volume"]].values
    else:
        result["volume"] = 0

    # Sort by timestamp ascending
    result = result.sort_index()

    return result


def scan_data_directory(directory: Path) -> list[Path]:
    """Scan a directory for parquet files.

    Returns a sorted list of Path objects for all .parquet files.
    Raises FileNotFoundError if directory doesn't exist.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    return sorted(directory.glob("*.parquet"))

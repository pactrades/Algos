"""Tests for data loader — parquet file loading, schema detection, OHLCV parsing."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from algos.core.data_loader import (
    detect_schema,
    load_parquet,
    normalize_ohlcv,
    scan_data_directory,
)


@pytest.fixture
def sample_parquet(tmp_path: Path) -> Path:
    """Create a sample parquet file with standard OHLCV columns."""
    n = 100
    np.random.seed(42)
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="5min", tz="US/Eastern")
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": 5000.0 + np.random.randn(n) * 2,
            "high": 5005.0 + np.abs(np.random.randn(n)) * 3,
            "low": 4995.0 - np.abs(np.random.randn(n)) * 3,
            "close": 5000.0 + np.cumsum(np.random.randn(n) * 0.5),
            "volume": np.random.randint(100, 10000, n),
        }
    )
    path = tmp_path / "ES_5m.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def uppercase_parquet(tmp_path: Path) -> Path:
    """Create a parquet file with uppercase column names."""
    n = 50
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-02", periods=n, freq="5min"),
            "Open": np.random.randn(n) + 5000,
            "High": np.random.randn(n) + 5005,
            "Low": np.random.randn(n) + 4995,
            "Close": np.random.randn(n) + 5000,
            "Volume": np.random.randint(100, 5000, n),
        }
    )
    path = tmp_path / "NQ_5m.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def nonstandard_parquet(tmp_path: Path) -> Path:
    """Create a parquet file with non-standard column names."""
    n = 50
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=n, freq="5min"),
            "o": np.random.randn(n) + 75,
            "h": np.random.randn(n) + 76,
            "l": np.random.randn(n) + 74,
            "c": np.random.randn(n) + 75,
            "vol": np.random.randint(100, 5000, n),
        }
    )
    path = tmp_path / "CL_5m.parquet"
    df.to_parquet(path, index=False)
    return path


@pytest.fixture
def data_directory(tmp_path: Path) -> Path:
    """Create a directory with multiple parquet files."""
    np.random.seed(42)
    for symbol in ["ES", "CL", "GC"]:
        n = 50
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-02", periods=n, freq="5min"),
                "open": np.random.randn(n) + 5000,
                "high": np.random.randn(n) + 5005,
                "low": np.random.randn(n) + 4995,
                "close": np.random.randn(n) + 5000,
                "volume": np.random.randint(100, 5000, n),
            }
        )
        df.to_parquet(tmp_path / f"{symbol}_5m.parquet", index=False)
    # Also put a non-parquet file to ensure it's skipped
    (tmp_path / "notes.txt").write_text("not a parquet file")
    return tmp_path


class TestLoadParquet:
    """Test raw parquet file loading."""

    def test_load_returns_dataframe(self, sample_parquet: Path) -> None:
        df = load_parquet(sample_parquet)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_parquet(Path("/nonexistent/file.parquet"))

    def test_load_preserves_all_columns(self, sample_parquet: Path) -> None:
        df = load_parquet(sample_parquet)
        assert "timestamp" in df.columns or "timestamp" in df.index.names
        assert len(df.columns) >= 5  # OHLCV at minimum


class TestDetectSchema:
    """Test automatic schema detection for different column naming conventions."""

    def test_detect_standard_lowercase(self, sample_parquet: Path) -> None:
        df = load_parquet(sample_parquet)
        schema = detect_schema(df)
        assert schema["timestamp"] == "timestamp"
        assert schema["open"] == "open"
        assert schema["high"] == "high"
        assert schema["low"] == "low"
        assert schema["close"] == "close"
        assert schema["volume"] == "volume"

    def test_detect_uppercase_columns(self, uppercase_parquet: Path) -> None:
        df = load_parquet(uppercase_parquet)
        schema = detect_schema(df)
        assert schema["timestamp"] == "Timestamp"
        assert schema["open"] == "Open"
        assert schema["close"] == "Close"

    def test_detect_abbreviated_columns(self, nonstandard_parquet: Path) -> None:
        df = load_parquet(nonstandard_parquet)
        schema = detect_schema(df)
        assert schema["timestamp"] == "date"
        assert schema["open"] == "o"
        assert schema["close"] == "c"
        assert schema["volume"] == "vol"

    def test_detect_raises_on_missing_ohlc(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "bad.parquet"
        df.to_parquet(path, index=False)
        loaded = load_parquet(path)
        with pytest.raises(ValueError, match="Could not detect"):
            detect_schema(loaded)


class TestNormalizeOHLCV:
    """Test normalization to standard OHLCV format."""

    def test_normalize_standard_columns(self, sample_parquet: Path) -> None:
        df = load_parquet(sample_parquet)
        normalized = normalize_ohlcv(df)
        assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(normalized.index, pd.DatetimeIndex)
        assert normalized.index.name == "timestamp"

    def test_normalize_uppercase_columns(self, uppercase_parquet: Path) -> None:
        df = load_parquet(uppercase_parquet)
        normalized = normalize_ohlcv(df)
        assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(normalized.index, pd.DatetimeIndex)

    def test_normalize_abbreviated_columns(self, nonstandard_parquet: Path) -> None:
        df = load_parquet(nonstandard_parquet)
        normalized = normalize_ohlcv(df)
        assert list(normalized.columns) == ["open", "high", "low", "close", "volume"]

    def test_normalize_sorts_by_timestamp(self, tmp_path: Path) -> None:
        """Ensure output is sorted ascending by timestamp."""
        dates = pd.date_range("2024-01-02", periods=10, freq="5min")
        df = pd.DataFrame(
            {
                "timestamp": dates[::-1],  # Reversed
                "open": range(10),
                "high": range(10),
                "low": range(10),
                "close": range(10),
                "volume": range(10),
            }
        )
        path = tmp_path / "reversed.parquet"
        df.to_parquet(path, index=False)
        normalized = normalize_ohlcv(load_parquet(path))
        assert normalized.index.is_monotonic_increasing

    def test_normalize_ohlcv_types_are_float(self, sample_parquet: Path) -> None:
        df = load_parquet(sample_parquet)
        normalized = normalize_ohlcv(df)
        for col in ["open", "high", "low", "close"]:
            assert normalized[col].dtype == np.float64
        assert pd.api.types.is_integer_dtype(normalized["volume"]) or pd.api.types.is_float_dtype(
            normalized["volume"]
        )


class TestScanDataDirectory:
    """Test directory scanning for parquet files."""

    def test_scan_finds_all_parquet_files(self, data_directory: Path) -> None:
        files = scan_data_directory(data_directory)
        assert len(files) == 3
        names = {f.stem for f in files}
        assert names == {"ES_5m", "CL_5m", "GC_5m"}

    def test_scan_ignores_non_parquet(self, data_directory: Path) -> None:
        files = scan_data_directory(data_directory)
        extensions = {f.suffix for f in files}
        assert extensions == {".parquet"}

    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        files = scan_data_directory(tmp_path)
        assert files == []

    def test_scan_nonexistent_directory_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            scan_data_directory(Path("/nonexistent/dir"))

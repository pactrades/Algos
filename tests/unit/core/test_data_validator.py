"""Tests for data validation — gap detection, outlier filtering, integrity checks."""

import numpy as np
import pandas as pd
import pytest

from algos.core.data_validator import (
    detect_gaps,
    detect_outliers,
    validate_ohlcv_integrity,
)


@pytest.fixture
def clean_ohlcv() -> pd.DataFrame:
    """Generate clean OHLCV data with no issues — proper bar structure."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="5min", tz="US/Eastern")
    close = 5000.0 + np.cumsum(np.random.randn(n) * 0.5)
    open_ = close + np.random.randn(n) * 0.3
    # Ensure high >= max(open, close) and low <= min(open, close)
    bar_max = np.maximum(open_, close)
    bar_min = np.minimum(open_, close)
    high = bar_max + np.abs(np.random.randn(n)) * 1.0
    low = bar_min - np.abs(np.random.randn(n)) * 1.0
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(100, 5000, n),
        },
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


@pytest.fixture
def gapped_ohlcv() -> pd.DataFrame:
    """Generate OHLCV data with gaps (missing bars)."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-02 09:30", periods=100, freq="5min", tz="US/Eastern")
    # Remove some bars to create gaps
    dates_with_gaps = dates.delete([10, 11, 12, 50, 51])
    n = len(dates_with_gaps)
    close = 5000.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.3,
            "high": close + np.abs(np.random.randn(n)) * 1.0,
            "low": close - np.abs(np.random.randn(n)) * 1.0,
            "close": close,
            "volume": np.random.randint(100, 5000, n),
        },
        index=pd.DatetimeIndex(dates_with_gaps, name="timestamp"),
    )


@pytest.fixture
def ohlcv_with_outliers() -> pd.DataFrame:
    """Generate OHLCV data with price outliers (spikes)."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="5min", tz="US/Eastern")
    close = 5000.0 + np.cumsum(np.random.randn(n) * 0.5)
    close[25] = 6000.0  # Extreme spike
    close[75] = 4000.0  # Extreme drop
    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.3,
            "high": close + np.abs(np.random.randn(n)) * 1.0,
            "low": close - np.abs(np.random.randn(n)) * 1.0,
            "close": close,
            "volume": np.random.randint(100, 5000, n),
        },
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


class TestDetectGaps:
    """Test gap detection in time series."""

    def test_no_gaps_in_clean_data(self, clean_ohlcv: pd.DataFrame) -> None:
        gaps = detect_gaps(clean_ohlcv, expected_freq="5min")
        assert len(gaps) == 0

    def test_detects_gaps_in_gapped_data(self, gapped_ohlcv: pd.DataFrame) -> None:
        gaps = detect_gaps(gapped_ohlcv, expected_freq="5min")
        assert len(gaps) == 2  # Two gap regions

    def test_gap_contains_start_and_end_timestamps(self, gapped_ohlcv: pd.DataFrame) -> None:
        gaps = detect_gaps(gapped_ohlcv, expected_freq="5min")
        for gap in gaps:
            assert "start" in gap
            assert "end" in gap
            assert "missing_bars" in gap

    def test_gap_missing_bar_count(self, gapped_ohlcv: pd.DataFrame) -> None:
        gaps = detect_gaps(gapped_ohlcv, expected_freq="5min")
        total_missing = sum(g["missing_bars"] for g in gaps)
        assert total_missing == 5  # We removed 5 bars


class TestDetectOutliers:
    """Test outlier detection in price data."""

    def test_no_outliers_in_clean_data(self, clean_ohlcv: pd.DataFrame) -> None:
        outlier_indices = detect_outliers(clean_ohlcv, z_threshold=4.0)
        assert len(outlier_indices) == 0

    def test_detects_extreme_spikes(self, ohlcv_with_outliers: pd.DataFrame) -> None:
        outlier_indices = detect_outliers(ohlcv_with_outliers, z_threshold=4.0)
        assert len(outlier_indices) >= 1  # Should catch at least the extreme spike

    def test_returns_integer_indices(self, ohlcv_with_outliers: pd.DataFrame) -> None:
        outlier_indices = detect_outliers(ohlcv_with_outliers, z_threshold=4.0)
        for idx in outlier_indices:
            assert isinstance(idx, (int, np.integer))


class TestValidateOHLCVIntegrity:
    """Test OHLCV data integrity checks."""

    def test_clean_data_passes(self, clean_ohlcv: pd.DataFrame) -> None:
        report = validate_ohlcv_integrity(clean_ohlcv)
        assert report.is_valid

    def test_detects_high_below_low(self) -> None:
        """High should always be >= low."""
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [99.0],  # Below low — invalid
                "low": [101.0],
                "close": [100.0],
                "volume": [1000],
            },
            index=pd.DatetimeIndex(
                pd.date_range("2024-01-02 09:30", periods=1, freq="5min"), name="timestamp"
            ),
        )
        report = validate_ohlcv_integrity(df)
        assert not report.is_valid
        assert "high_below_low" in report.issues

    def test_detects_negative_volume(self) -> None:
        df = pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [-500],  # Negative — invalid
            },
            index=pd.DatetimeIndex(
                pd.date_range("2024-01-02 09:30", periods=1, freq="5min"), name="timestamp"
            ),
        )
        report = validate_ohlcv_integrity(df)
        assert not report.is_valid
        assert "negative_volume" in report.issues

    def test_detects_nan_values(self) -> None:
        df = pd.DataFrame(
            {
                "open": [100.0, float("nan")],
                "high": [101.0, 102.0],
                "low": [99.0, 98.0],
                "close": [100.0, 101.0],
                "volume": [1000, 2000],
            },
            index=pd.DatetimeIndex(
                pd.date_range("2024-01-02 09:30", periods=2, freq="5min"), name="timestamp"
            ),
        )
        report = validate_ohlcv_integrity(df)
        assert not report.is_valid
        assert "nan_values" in report.issues

    def test_detects_open_outside_high_low(self) -> None:
        """Open should be between low and high."""
        df = pd.DataFrame(
            {
                "open": [110.0],  # Above high — invalid
                "high": [105.0],
                "low": [95.0],
                "close": [100.0],
                "volume": [1000],
            },
            index=pd.DatetimeIndex(
                pd.date_range("2024-01-02 09:30", periods=1, freq="5min"), name="timestamp"
            ),
        )
        report = validate_ohlcv_integrity(df)
        assert not report.is_valid
        assert "open_outside_range" in report.issues

    def test_report_has_row_count(self, clean_ohlcv: pd.DataFrame) -> None:
        report = validate_ohlcv_integrity(clean_ohlcv)
        assert report.row_count == 100

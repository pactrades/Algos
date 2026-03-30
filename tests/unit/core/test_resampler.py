"""Tests for timeframe resampling — converting OHLCV between timeframes."""

import numpy as np
import pandas as pd
import pytest

from algos.core.resampler import VALID_TIMEFRAMES, resample_ohlcv


@pytest.fixture
def minute_data() -> pd.DataFrame:
    """Generate 1-minute OHLCV data for one trading day."""
    np.random.seed(42)
    n = 390  # 6.5 hours of 1-min bars (RTH)
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="US/Eastern")
    close = 5000.0 + np.cumsum(np.random.randn(n) * 0.25)
    high = close + np.abs(np.random.randn(n)) * 0.5
    low = close - np.abs(np.random.randn(n)) * 0.5
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.randint(50, 5000, n)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


class TestResampleOHLCV:
    """Test OHLCV timeframe resampling."""

    def test_1m_to_5m_bar_count(self, minute_data: pd.DataFrame) -> None:
        result = resample_ohlcv(minute_data, "5m")
        assert len(result) == 78  # 390 / 5

    def test_1m_to_15m_bar_count(self, minute_data: pd.DataFrame) -> None:
        result = resample_ohlcv(minute_data, "15m")
        assert len(result) == 26  # 390 / 15

    def test_1m_to_1h_bar_count(self, minute_data: pd.DataFrame) -> None:
        result = resample_ohlcv(minute_data, "1h")
        # 390 min = 6h30m → 7 bars (6 full hours + 1 partial)
        assert len(result) == 7

    def test_ohlcv_aggregation_correct(self, minute_data: pd.DataFrame) -> None:
        """Verify OHLCV aggregation rules: first open, max high, min low, last close, sum volume."""
        result = resample_ohlcv(minute_data, "5m")
        first_5_bars = minute_data.iloc[:5]
        first_resampled = result.iloc[0]
        assert first_resampled["open"] == pytest.approx(first_5_bars["open"].iloc[0])
        assert first_resampled["high"] == pytest.approx(first_5_bars["high"].max())
        assert first_resampled["low"] == pytest.approx(first_5_bars["low"].min())
        assert first_resampled["close"] == pytest.approx(first_5_bars["close"].iloc[-1])
        assert first_resampled["volume"] == first_5_bars["volume"].sum()

    def test_output_columns_match_input(self, minute_data: pd.DataFrame) -> None:
        result = resample_ohlcv(minute_data, "5m")
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    def test_output_index_is_datetime(self, minute_data: pd.DataFrame) -> None:
        result = resample_ohlcv(minute_data, "5m")
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "timestamp"

    def test_output_sorted_ascending(self, minute_data: pd.DataFrame) -> None:
        result = resample_ohlcv(minute_data, "5m")
        assert result.index.is_monotonic_increasing

    def test_no_nan_in_output(self, minute_data: pd.DataFrame) -> None:
        result = resample_ohlcv(minute_data, "5m")
        assert not result.isna().any().any()

    def test_invalid_timeframe_raises(self, minute_data: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Invalid timeframe"):
            resample_ohlcv(minute_data, "7m")

    def test_same_timeframe_returns_copy(self, minute_data: pd.DataFrame) -> None:
        result = resample_ohlcv(minute_data, "1m")
        assert len(result) == len(minute_data)
        assert result is not minute_data  # Should be a copy


class TestValidTimeframes:
    """Test the valid timeframes constant."""

    def test_includes_common_timeframes(self) -> None:
        for tf in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
            assert tf in VALID_TIMEFRAMES

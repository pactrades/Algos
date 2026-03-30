"""Tests for multi-instrument data alignment — synchronizing timestamps."""

import numpy as np
import pandas as pd
import pytest

from algos.core.alignment import align_instruments, find_common_timerange


@pytest.fixture
def overlapping_data() -> dict[str, pd.DataFrame]:
    """Two instruments with overlapping but not identical timestamps."""
    np.random.seed(42)
    dates_es = pd.date_range("2024-01-02 09:30", periods=100, freq="5min", tz="US/Eastern")
    dates_cl = pd.date_range("2024-01-02 09:35", periods=95, freq="5min", tz="US/Eastern")

    def make_df(dates: pd.DatetimeIndex) -> pd.DataFrame:
        n = len(dates)
        close = 5000.0 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame(
            {
                "open": close + np.random.randn(n) * 0.3,
                "high": close + np.abs(np.random.randn(n)),
                "low": close - np.abs(np.random.randn(n)),
                "close": close,
                "volume": np.random.randint(100, 5000, n),
            },
            index=pd.DatetimeIndex(dates, name="timestamp"),
        )

    return {"ES": make_df(dates_es), "CL": make_df(dates_cl)}


@pytest.fixture
def identical_data() -> dict[str, pd.DataFrame]:
    """Two instruments with perfectly aligned timestamps."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-02 09:30", periods=100, freq="5min", tz="US/Eastern")

    def make_df(base: float) -> pd.DataFrame:
        n = 100
        close = base + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame(
            {
                "open": close + np.random.randn(n) * 0.3,
                "high": close + np.abs(np.random.randn(n)),
                "low": close - np.abs(np.random.randn(n)),
                "close": close,
                "volume": np.random.randint(100, 5000, n),
            },
            index=pd.DatetimeIndex(dates, name="timestamp"),
        )

    return {"ES": make_df(5000.0), "GC": make_df(2050.0)}


class TestAlignInstruments:
    """Test multi-instrument timestamp alignment."""

    def test_aligned_data_has_same_timestamps(self, overlapping_data: dict) -> None:
        aligned = align_instruments(overlapping_data)
        timestamps_es = aligned["ES"].index
        timestamps_cl = aligned["CL"].index
        pd.testing.assert_index_equal(timestamps_es, timestamps_cl)

    def test_aligned_data_is_inner_join(self, overlapping_data: dict) -> None:
        """Only timestamps present in ALL instruments should remain."""
        aligned = align_instruments(overlapping_data)
        for df in aligned.values():
            assert len(df) > 0
            # Should be <= min of original lengths
            min_original = min(len(v) for v in overlapping_data.values())
            assert len(df) <= min_original

    def test_identical_data_unchanged(self, identical_data: dict) -> None:
        aligned = align_instruments(identical_data)
        for symbol in identical_data:
            assert len(aligned[symbol]) == len(identical_data[symbol])

    def test_preserves_ohlcv_columns(self, overlapping_data: dict) -> None:
        aligned = align_instruments(overlapping_data)
        for df in aligned.values():
            assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_empty_input_returns_empty(self) -> None:
        aligned = align_instruments({})
        assert aligned == {}

    def test_single_instrument_unchanged(self, identical_data: dict) -> None:
        single = {"ES": identical_data["ES"]}
        aligned = align_instruments(single)
        assert len(aligned["ES"]) == len(single["ES"])


class TestFindCommonTimerange:
    """Test finding the common time range across instruments."""

    def test_common_range_with_overlap(self, overlapping_data: dict) -> None:
        start, end = find_common_timerange(overlapping_data)
        # Start should be the max of all starts
        max_start = max(df.index.min() for df in overlapping_data.values())
        min_end = min(df.index.max() for df in overlapping_data.values())
        assert start == max_start
        assert end == min_end

    def test_common_range_identical_data(self, identical_data: dict) -> None:
        start, end = find_common_timerange(identical_data)
        assert start == identical_data["ES"].index.min()
        assert end == identical_data["ES"].index.max()

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            find_common_timerange({})

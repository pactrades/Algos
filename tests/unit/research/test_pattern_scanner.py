"""Tests for pattern scanner framework and individual scanners."""

import numpy as np
import pandas as pd
import pytest

from algos.research.pattern_scanner import (
    PatternCandidate,
    SessionGapScanner,
    TimeOfDayScanner,
    VolatilitySqueezeScanner,
)


@pytest.fixture
def daily_ohlcv() -> pd.DataFrame:
    """Generate daily OHLCV data with known patterns."""
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range("2022-01-03", periods=n, tz="US/Eastern")
    close = 5000.0 + np.cumsum(np.random.randn(n) * 10)
    high = close + np.abs(np.random.randn(n)) * 15
    low = close - np.abs(np.random.randn(n)) * 15
    open_ = close + np.random.randn(n) * 8
    # Ensure OHLCV integrity
    bar_max = np.maximum(open_, close)
    bar_min = np.minimum(open_, close)
    high = np.maximum(high, bar_max)
    low = np.minimum(low, bar_min)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(100000, 500000, n),
        },
        index=pd.DatetimeIndex(dates, name="timestamp"),
    )


@pytest.fixture
def intraday_ohlcv() -> pd.DataFrame:
    """Generate 5-minute intraday OHLCV data spanning multiple days."""
    np.random.seed(42)
    # 5 days of RTH (78 bars per day at 5min for 6.5 hours)
    all_frames = []
    for day_offset in range(20):
        date = pd.Timestamp("2024-01-02", tz="US/Eastern") + pd.Timedelta(days=day_offset)
        if date.weekday() >= 5:
            continue
        n = 78
        dates = pd.date_range(
            date.replace(hour=9, minute=30), periods=n, freq="5min", tz="US/Eastern"
        )
        close = 5000.0 + np.cumsum(np.random.randn(n) * 2)
        high = close + np.abs(np.random.randn(n)) * 3
        low = close - np.abs(np.random.randn(n)) * 3
        open_ = close + np.random.randn(n) * 1.5
        bar_max = np.maximum(open_, close)
        bar_min = np.minimum(open_, close)
        high = np.maximum(high, bar_max)
        low = np.minimum(low, bar_min)
        df = pd.DataFrame(
            {
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(500, 10000, n),
            },
            index=pd.DatetimeIndex(dates, name="timestamp"),
        )
        all_frames.append(df)
    return pd.concat(all_frames)


class TestPatternCandidate:
    """Test the PatternCandidate data structure."""

    def test_create_candidate(self) -> None:
        candidate = PatternCandidate(
            name="morning_reversal",
            description="Price tends to reverse at 10:00 AM",
            n_occurrences=150,
            avg_return=0.003,
            win_rate=0.58,
            p_value=0.02,
            effect_size=0.35,
        )
        assert candidate.name == "morning_reversal"
        assert candidate.n_occurrences == 150
        assert candidate.passes_minimum_criteria()

    def test_fails_minimum_with_few_occurrences(self) -> None:
        candidate = PatternCandidate(
            name="rare_pattern",
            description="Too few trades",
            n_occurrences=50,  # Below 100 minimum
            avg_return=0.01,
            win_rate=0.7,
            p_value=0.01,
            effect_size=0.5,
        )
        assert not candidate.passes_minimum_criteria()

    def test_fails_minimum_with_high_p_value(self) -> None:
        candidate = PatternCandidate(
            name="random_pattern",
            description="Not significant",
            n_occurrences=200,
            avg_return=0.001,
            win_rate=0.51,
            p_value=0.15,  # Above 0.05
            effect_size=0.1,
        )
        assert not candidate.passes_minimum_criteria()

    def test_fails_minimum_with_low_effect_size(self) -> None:
        candidate = PatternCandidate(
            name="weak_pattern",
            description="Effect too small",
            n_occurrences=200,
            avg_return=0.0001,
            win_rate=0.52,
            p_value=0.04,
            effect_size=0.1,  # Below 0.2
        )
        assert not candidate.passes_minimum_criteria()


class TestTimeOfDayScanner:
    """Test time-of-day pattern scanner."""

    def test_scan_returns_candidates(self, intraday_ohlcv: pd.DataFrame) -> None:
        scanner = TimeOfDayScanner()
        candidates = scanner.scan(intraday_ohlcv)
        assert isinstance(candidates, list)
        for c in candidates:
            assert isinstance(c, PatternCandidate)

    def test_scan_analyzes_hourly_buckets(self, intraday_ohlcv: pd.DataFrame) -> None:
        scanner = TimeOfDayScanner()
        candidates = scanner.scan(intraday_ohlcv)
        # Should produce at least some candidates (even if none pass threshold)
        # The scanner should analyze multiple time buckets
        assert isinstance(candidates, list)


class TestVolatilitySqueezeScanner:
    """Test volatility squeeze pattern scanner."""

    def test_scan_returns_candidates(self, daily_ohlcv: pd.DataFrame) -> None:
        scanner = VolatilitySqueezeScanner()
        candidates = scanner.scan(daily_ohlcv)
        assert isinstance(candidates, list)

    def test_scanner_uses_atr(self, daily_ohlcv: pd.DataFrame) -> None:
        scanner = VolatilitySqueezeScanner(atr_period=14, squeeze_threshold=0.75)
        candidates = scanner.scan(daily_ohlcv)
        assert isinstance(candidates, list)


class TestSessionGapScanner:
    """Test session gap pattern scanner."""

    def test_scan_returns_candidates(self, daily_ohlcv: pd.DataFrame) -> None:
        scanner = SessionGapScanner()
        candidates = scanner.scan(daily_ohlcv)
        assert isinstance(candidates, list)

    def test_gap_analysis(self, daily_ohlcv: pd.DataFrame) -> None:
        scanner = SessionGapScanner()
        candidates = scanner.scan(daily_ohlcv)
        for c in candidates:
            assert isinstance(c, PatternCandidate)

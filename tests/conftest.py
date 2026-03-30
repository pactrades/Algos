"""Shared test fixtures and sample data generators."""

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from algos.core.signal import Signal


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate a simple OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="5min", tz="US/Eastern")
    close = 5000.0 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n)) * 3
    low = close - np.abs(np.random.randn(n)) * 3
    open_ = close + np.random.randn(n) * 1.5

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(100, 10000, n),
        }
    ).set_index("timestamp")


@pytest.fixture
def sample_signal() -> Signal:
    """Create a sample Signal for testing."""
    return Signal(
        strategy="temporal_edge",
        symbol="ES",
        micro_symbol="MES",
        direction="LONG",
        entry_price=5000.00,
        stop_loss=4990.00,
        take_profit=5020.00,
        contracts=2,
        risk_dollars=25.00,
        timeframe="5m",
        timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=UTC),
        expiry=datetime(2024, 1, 15, 11, 0, tzinfo=UTC),
        confidence=0.75,
        metadata={"pattern": "session_open_drive", "p_value": 0.02},
    )


@pytest.fixture
def multi_instrument_ohlcv() -> dict[str, pd.DataFrame]:
    """Generate OHLCV data for multiple instruments."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="5min", tz="US/Eastern")

    instruments: dict[str, dict[str, Any]] = {
        "ES": {"base": 5000.0, "vol": 2.0},
        "NQ": {"base": 17500.0, "vol": 8.0},
        "CL": {"base": 75.0, "vol": 0.3},
        "GC": {"base": 2050.0, "vol": 3.0},
        "ZB": {"base": 118.0, "vol": 0.2},
    }

    result = {}
    for symbol, params in instruments.items():
        close = params["base"] + np.cumsum(np.random.randn(n) * params["vol"])
        high = close + np.abs(np.random.randn(n)) * params["vol"] * 1.5
        low = close - np.abs(np.random.randn(n)) * params["vol"] * 1.5
        open_ = close + np.random.randn(n) * params["vol"] * 0.75

        result[symbol] = pd.DataFrame(
            {
                "timestamp": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.random.randint(100, 10000, n),
            }
        ).set_index("timestamp")

    return result

"""Tests for the Signal model."""

import json
from datetime import UTC, datetime

import pytest

from algos.core.signal import Signal


class TestSignalCreation:
    """Test Signal model creation and validation."""

    def test_create_valid_long_signal(self, sample_signal: Signal) -> None:
        assert sample_signal.strategy == "temporal_edge"
        assert sample_signal.symbol == "ES"
        assert sample_signal.micro_symbol == "MES"
        assert sample_signal.direction == "LONG"
        assert sample_signal.entry_price == 5000.00
        assert sample_signal.stop_loss == 4990.00
        assert sample_signal.take_profit == 5020.00
        assert sample_signal.contracts == 2
        assert sample_signal.risk_dollars == 25.00
        assert sample_signal.confidence == 0.75

    def test_create_valid_short_signal(self) -> None:
        signal = Signal(
            strategy="vol_regime",
            symbol="CL",
            micro_symbol="MCL",
            direction="SHORT",
            entry_price=75.00,
            stop_loss=76.00,
            take_profit=73.00,
            contracts=1,
            risk_dollars=100.00,
            timeframe="1h",
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=UTC),
            confidence=0.80,
        )
        assert signal.direction == "SHORT"
        assert signal.stop_loss > signal.entry_price
        assert signal.take_profit < signal.entry_price

    def test_signal_without_take_profit(self) -> None:
        signal = Signal(
            strategy="temporal_edge",
            symbol="ES",
            micro_symbol="MES",
            direction="LONG",
            entry_price=5000.00,
            stop_loss=4990.00,
            take_profit=None,
            contracts=1,
            risk_dollars=12.50,
            timeframe="5m",
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=UTC),
            confidence=0.60,
        )
        assert signal.take_profit is None

    def test_signal_without_expiry(self) -> None:
        signal = Signal(
            strategy="temporal_edge",
            symbol="ES",
            micro_symbol="MES",
            direction="LONG",
            entry_price=5000.00,
            stop_loss=4990.00,
            contracts=1,
            risk_dollars=12.50,
            timeframe="5m",
            timestamp=datetime(2024, 1, 15, 10, 30, tzinfo=UTC),
            confidence=0.60,
        )
        assert signal.expiry is None


class TestSignalValidation:
    """Test Signal model validation rules."""

    def test_long_stop_loss_must_be_below_entry(self) -> None:
        with pytest.raises(ValueError, match="Stop loss must be below entry for LONG"):
            Signal(
                strategy="test",
                symbol="ES",
                micro_symbol="MES",
                direction="LONG",
                entry_price=5000.00,
                stop_loss=5010.00,  # Above entry — invalid for LONG
                contracts=1,
                risk_dollars=12.50,
                timeframe="5m",
                timestamp=datetime(2024, 1, 15, tzinfo=UTC),
                confidence=0.5,
            )

    def test_short_stop_loss_must_be_above_entry(self) -> None:
        with pytest.raises(ValueError, match="Stop loss must be above entry for SHORT"):
            Signal(
                strategy="test",
                symbol="CL",
                micro_symbol="MCL",
                direction="SHORT",
                entry_price=75.00,
                stop_loss=74.00,  # Below entry — invalid for SHORT
                contracts=1,
                risk_dollars=10.00,
                timeframe="1h",
                timestamp=datetime(2024, 1, 15, tzinfo=UTC),
                confidence=0.5,
            )

    def test_long_take_profit_must_be_above_entry(self) -> None:
        with pytest.raises(ValueError, match="Take profit must be above entry for LONG"):
            Signal(
                strategy="test",
                symbol="ES",
                micro_symbol="MES",
                direction="LONG",
                entry_price=5000.00,
                stop_loss=4990.00,
                take_profit=4995.00,  # Below entry — invalid for LONG
                contracts=1,
                risk_dollars=12.50,
                timeframe="5m",
                timestamp=datetime(2024, 1, 15, tzinfo=UTC),
                confidence=0.5,
            )

    def test_short_take_profit_must_be_below_entry(self) -> None:
        with pytest.raises(ValueError, match="Take profit must be below entry for SHORT"):
            Signal(
                strategy="test",
                symbol="CL",
                micro_symbol="MCL",
                direction="SHORT",
                entry_price=75.00,
                stop_loss=76.00,
                take_profit=76.50,  # Above entry — invalid for SHORT
                contracts=1,
                risk_dollars=10.00,
                timeframe="1h",
                timestamp=datetime(2024, 1, 15, tzinfo=UTC),
                confidence=0.5,
            )

    def test_entry_price_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            Signal(
                strategy="test",
                symbol="ES",
                micro_symbol="MES",
                direction="LONG",
                entry_price=-100.0,
                stop_loss=4990.00,
                contracts=1,
                risk_dollars=12.50,
                timeframe="5m",
                timestamp=datetime(2024, 1, 15, tzinfo=UTC),
                confidence=0.5,
            )

    def test_confidence_must_be_between_0_and_1(self) -> None:
        with pytest.raises(ValueError):
            Signal(
                strategy="test",
                symbol="ES",
                micro_symbol="MES",
                direction="LONG",
                entry_price=5000.00,
                stop_loss=4990.00,
                contracts=1,
                risk_dollars=12.50,
                timeframe="5m",
                timestamp=datetime(2024, 1, 15, tzinfo=UTC),
                confidence=1.5,  # Out of range
            )

    def test_contracts_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            Signal(
                strategy="test",
                symbol="ES",
                micro_symbol="MES",
                direction="LONG",
                entry_price=5000.00,
                stop_loss=4990.00,
                contracts=0,
                risk_dollars=12.50,
                timeframe="5m",
                timestamp=datetime(2024, 1, 15, tzinfo=UTC),
                confidence=0.5,
            )


class TestSignalSerialization:
    """Test Signal JSON serialization."""

    def test_to_json_returns_valid_json(self, sample_signal: Signal) -> None:
        json_str = sample_signal.to_json()
        parsed = json.loads(json_str)
        assert parsed["strategy"] == "temporal_edge"
        assert parsed["symbol"] == "ES"
        assert parsed["direction"] == "LONG"
        assert parsed["entry_price"] == 5000.00

    def test_roundtrip_serialization(self, sample_signal: Signal) -> None:
        json_str = sample_signal.to_json()
        restored = Signal.model_validate_json(json_str)
        assert restored == sample_signal

    def test_metadata_included_in_json(self, sample_signal: Signal) -> None:
        json_str = sample_signal.to_json()
        parsed = json.loads(json_str)
        assert parsed["metadata"]["pattern"] == "session_open_drive"
        assert parsed["metadata"]["p_value"] == 0.02


class TestSignalRiskReward:
    """Test risk:reward ratio calculation."""

    def test_risk_reward_ratio_long(self, sample_signal: Signal) -> None:
        # Entry=5000, SL=4990, TP=5020 → risk=10, reward=20 → R:R=2.0
        rr = sample_signal.risk_reward_ratio()
        assert rr == pytest.approx(2.0)

    def test_risk_reward_ratio_short(self) -> None:
        signal = Signal(
            strategy="test",
            symbol="CL",
            micro_symbol="MCL",
            direction="SHORT",
            entry_price=75.00,
            stop_loss=76.00,
            take_profit=73.00,
            contracts=1,
            risk_dollars=10.00,
            timeframe="1h",
            timestamp=datetime(2024, 1, 15, tzinfo=UTC),
            confidence=0.5,
        )
        rr = signal.risk_reward_ratio()
        assert rr == pytest.approx(2.0)

    def test_risk_reward_ratio_none_without_take_profit(self) -> None:
        signal = Signal(
            strategy="test",
            symbol="ES",
            micro_symbol="MES",
            direction="LONG",
            entry_price=5000.00,
            stop_loss=4990.00,
            take_profit=None,
            contracts=1,
            risk_dollars=12.50,
            timeframe="5m",
            timestamp=datetime(2024, 1, 15, tzinfo=UTC),
            confidence=0.5,
        )
        assert signal.risk_reward_ratio() is None

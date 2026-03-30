"""Tests for CME session time utilities and NQ blackout window."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from algos.core.session import (
    get_instrument_type,
    is_eth,
    is_in_nq_blackout,
    is_rth,
    should_suppress_nq_signal,
    to_eastern,
)

ET = ZoneInfo("US/Eastern")


class TestToEastern:
    """Test timezone conversion."""

    def test_utc_to_eastern(self) -> None:
        utc_dt = datetime(2024, 1, 15, 15, 30, tzinfo=UTC)
        et_dt = to_eastern(utc_dt)
        assert et_dt.tzinfo is not None
        assert et_dt.hour == 10  # UTC-5 in January
        assert et_dt.minute == 30

    def test_naive_datetime_treated_as_utc(self) -> None:
        naive_dt = datetime(2024, 1, 15, 15, 30)
        et_dt = to_eastern(naive_dt)
        assert et_dt.hour == 10


class TestNQBlackout:
    """Test NQ blackout window (9:30 AM - 12:00 PM ET)."""

    def test_inside_blackout_930am(self) -> None:
        dt = datetime(2024, 1, 15, 9, 30, tzinfo=ET)
        assert is_in_nq_blackout(dt) is True

    def test_inside_blackout_1100am(self) -> None:
        dt = datetime(2024, 1, 15, 11, 0, tzinfo=ET)
        assert is_in_nq_blackout(dt) is True

    def test_outside_blackout_1200pm(self) -> None:
        dt = datetime(2024, 1, 15, 12, 0, tzinfo=ET)
        assert is_in_nq_blackout(dt) is False

    def test_outside_blackout_929am(self) -> None:
        dt = datetime(2024, 1, 15, 9, 29, tzinfo=ET)
        assert is_in_nq_blackout(dt) is False

    def test_outside_blackout_afternoon(self) -> None:
        dt = datetime(2024, 1, 15, 14, 0, tzinfo=ET)
        assert is_in_nq_blackout(dt) is False

    def test_blackout_with_utc_input(self) -> None:
        # 10:00 AM ET = 3:00 PM UTC in January
        utc_dt = datetime(2024, 1, 15, 15, 0, tzinfo=UTC)
        assert is_in_nq_blackout(utc_dt) is True


class TestShouldSuppressNQSignal:
    """Test NQ signal suppression logic."""

    def test_suppress_nq_during_blackout(self) -> None:
        dt = datetime(2024, 1, 15, 10, 0, tzinfo=ET)
        assert should_suppress_nq_signal(dt, "NQ") is True

    def test_suppress_mnq_during_blackout(self) -> None:
        dt = datetime(2024, 1, 15, 10, 0, tzinfo=ET)
        assert should_suppress_nq_signal(dt, "MNQ") is True

    def test_allow_nq_outside_blackout(self) -> None:
        dt = datetime(2024, 1, 15, 14, 0, tzinfo=ET)
        assert should_suppress_nq_signal(dt, "NQ") is False

    def test_never_suppress_es(self) -> None:
        dt = datetime(2024, 1, 15, 10, 0, tzinfo=ET)
        assert should_suppress_nq_signal(dt, "ES") is False

    def test_never_suppress_cl(self) -> None:
        dt = datetime(2024, 1, 15, 10, 0, tzinfo=ET)
        assert should_suppress_nq_signal(dt, "CL") is False


class TestIsRTH:
    """Test Regular Trading Hours detection."""

    def test_equity_rth_open(self) -> None:
        dt = datetime(2024, 1, 15, 10, 0, tzinfo=ET)  # Monday 10AM
        assert is_rth(dt, "equity") is True

    def test_equity_before_rth(self) -> None:
        dt = datetime(2024, 1, 15, 9, 0, tzinfo=ET)
        assert is_rth(dt, "equity") is False

    def test_equity_after_rth(self) -> None:
        dt = datetime(2024, 1, 15, 16, 30, tzinfo=ET)
        assert is_rth(dt, "equity") is False

    def test_weekend_not_rth(self) -> None:
        dt = datetime(2024, 1, 13, 10, 0, tzinfo=ET)  # Saturday
        assert is_rth(dt, "equity") is False

    def test_energy_rth(self) -> None:
        dt = datetime(2024, 1, 15, 10, 0, tzinfo=ET)
        assert is_rth(dt, "energy") is True

    def test_energy_after_rth(self) -> None:
        dt = datetime(2024, 1, 15, 15, 0, tzinfo=ET)
        assert is_rth(dt, "energy") is False

    def test_metals_rth(self) -> None:
        dt = datetime(2024, 1, 15, 10, 0, tzinfo=ET)
        assert is_rth(dt, "metals") is True


class TestIsETH:
    """Test Electronic Trading Hours detection."""

    def test_weekday_open(self) -> None:
        dt = datetime(2024, 1, 15, 10, 0, tzinfo=ET)  # Monday 10AM
        assert is_eth(dt) is True

    def test_maintenance_halt(self) -> None:
        dt = datetime(2024, 1, 15, 17, 30, tzinfo=ET)  # 5:30 PM
        assert is_eth(dt) is False

    def test_after_maintenance(self) -> None:
        dt = datetime(2024, 1, 15, 18, 30, tzinfo=ET)  # 6:30 PM
        assert is_eth(dt) is True

    def test_saturday_closed(self) -> None:
        dt = datetime(2024, 1, 13, 10, 0, tzinfo=ET)
        assert is_eth(dt) is False

    def test_sunday_before_open(self) -> None:
        dt = datetime(2024, 1, 14, 15, 0, tzinfo=ET)  # Sunday 3PM
        assert is_eth(dt) is False

    def test_sunday_after_open(self) -> None:
        dt = datetime(2024, 1, 14, 19, 0, tzinfo=ET)  # Sunday 7PM
        assert is_eth(dt) is True

    def test_friday_before_close(self) -> None:
        dt = datetime(2024, 1, 12, 16, 0, tzinfo=ET)  # Friday 4PM
        assert is_eth(dt) is True

    def test_friday_after_close(self) -> None:
        dt = datetime(2024, 1, 12, 17, 30, tzinfo=ET)  # Friday 5:30PM
        assert is_eth(dt) is False


class TestGetInstrumentType:
    """Test instrument type mapping."""

    def test_equity_instruments(self) -> None:
        for sym in ["ES", "MES", "NQ", "MNQ", "YM", "MYM", "RTY", "M2K"]:
            assert get_instrument_type(sym) == "equity"

    def test_energy_instruments(self) -> None:
        for sym in ["CL", "MCL"]:
            assert get_instrument_type(sym) == "energy"

    def test_metals_instruments(self) -> None:
        for sym in ["GC", "MGC", "SI", "SIL"]:
            assert get_instrument_type(sym) == "metals"

    def test_bond_instruments(self) -> None:
        for sym in ["ZB", "ZN"]:
            assert get_instrument_type(sym) == "bonds"

    def test_unknown_defaults_to_equity(self) -> None:
        assert get_instrument_type("UNKNOWN") == "equity"

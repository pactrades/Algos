"""CME session time utilities and NQ blackout window enforcement."""

from __future__ import annotations

from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

ET = ZoneInfo("US/Eastern")

# NQ blackout window: user manually trades NQ from 9:30 AM - 12:00 PM ET
NQ_BLACKOUT_START = time(9, 30)
NQ_BLACKOUT_END = time(12, 0)

# CME equity index futures: Sunday 6PM ET - Friday 5PM ET, daily halt 5PM-6PM ET
CME_EQUITY_ETH_OPEN = time(18, 0)  # 6:00 PM ET
CME_EQUITY_ETH_CLOSE = time(17, 0)  # 5:00 PM ET (next day)
CME_EQUITY_RTH_OPEN = time(9, 30)
CME_EQUITY_RTH_CLOSE = time(16, 0)

# CME energy/metals: similar hours but different RTH
CME_ENERGY_RTH_OPEN = time(9, 0)
CME_ENERGY_RTH_CLOSE = time(14, 30)
CME_METALS_RTH_OPEN = time(8, 20)
CME_METALS_RTH_CLOSE = time(13, 30)


def to_eastern(dt: datetime) -> datetime:
    """Convert a datetime to US/Eastern timezone."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(ET)


def is_in_nq_blackout(dt: datetime) -> bool:
    """Check if a datetime falls within the NQ blackout window (9:30 AM - 12:00 PM ET).

    During this window, the user is manually trading NQ, so automated NQ signals
    should be suppressed.
    """
    et_time = to_eastern(dt).time()
    return NQ_BLACKOUT_START <= et_time < NQ_BLACKOUT_END


def is_rth(dt: datetime, instrument_type: str = "equity") -> bool:
    """Check if a datetime falls within Regular Trading Hours.

    Args:
        dt: The datetime to check.
        instrument_type: One of 'equity', 'energy', 'metals', 'bonds'.
    """
    et_time = to_eastern(dt).time()
    et_dt = to_eastern(dt)

    # No RTH on weekends
    if et_dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    rth_hours = {
        "equity": (CME_EQUITY_RTH_OPEN, CME_EQUITY_RTH_CLOSE),
        "energy": (CME_ENERGY_RTH_OPEN, CME_ENERGY_RTH_CLOSE),
        "metals": (CME_METALS_RTH_OPEN, CME_METALS_RTH_CLOSE),
        "bonds": (time(8, 20), time(15, 0)),
    }

    default = (CME_EQUITY_RTH_OPEN, CME_EQUITY_RTH_CLOSE)
    open_time, close_time = rth_hours.get(instrument_type, default)
    return open_time <= et_time < close_time


def is_eth(dt: datetime) -> bool:
    """Check if a datetime falls within Electronic Trading Hours.

    ETH runs from 6:00 PM ET to 5:00 PM ET next day (23 hours),
    Sunday through Friday. Daily maintenance halt from 5:00 PM - 6:00 PM ET.
    """
    et_dt = to_eastern(dt)
    et_time = et_dt.time()

    # Sunday after 6PM is open
    if et_dt.weekday() == 6:  # Sunday
        return et_time >= CME_EQUITY_ETH_OPEN

    # Friday before 5PM is open (closes at 5PM Friday)
    if et_dt.weekday() == 4:  # Friday
        return et_time < CME_EQUITY_ETH_CLOSE

    # Saturday is always closed
    if et_dt.weekday() == 5:
        return False

    # Mon-Thu: open except during 5PM-6PM maintenance
    return not (CME_EQUITY_ETH_CLOSE <= et_time < CME_EQUITY_ETH_OPEN)


def should_suppress_nq_signal(dt: datetime, symbol: str) -> bool:
    """Determine if a signal for a given symbol should be suppressed.

    Suppresses NQ/MNQ signals during the user's manual trading window.
    """
    if symbol.upper() in ("NQ", "MNQ"):
        return is_in_nq_blackout(dt)
    return False


INSTRUMENT_TYPE_MAP: dict[str, str] = {
    "ES": "equity",
    "MES": "equity",
    "NQ": "equity",
    "MNQ": "equity",
    "YM": "equity",
    "MYM": "equity",
    "RTY": "equity",
    "M2K": "equity",
    "CL": "energy",
    "MCL": "energy",
    "GC": "metals",
    "MGC": "metals",
    "SI": "metals",
    "SIL": "metals",
    "ZB": "bonds",
    "ZN": "bonds",
}


def get_instrument_type(symbol: str) -> str:
    """Get the instrument type for session hour lookups."""
    return INSTRUMENT_TYPE_MAP.get(symbol.upper(), "equity")

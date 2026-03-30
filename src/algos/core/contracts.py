"""Futures contract specifications — tick sizes, multipliers, trading hours."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractSpec:
    """Specification for a futures contract and its micro equivalent."""

    symbol: str
    micro_symbol: str
    name: str
    exchange: str
    tick_size: float
    tick_value: float  # Dollar value per tick for the MICRO contract
    full_tick_value: float  # Dollar value per tick for the full contract
    multiplier: int  # Micro-to-full ratio (e.g., 10 MES = 1 ES)
    rth_open: str  # Regular trading hours open (ET), e.g., "09:30"
    rth_close: str  # Regular trading hours close (ET), e.g., "16:00"
    eth_open: str  # Electronic trading hours open (ET), e.g., "18:00"
    eth_close: str  # Electronic trading hours close (ET), e.g., "17:00"

    def ticks_between(self, price_a: float, price_b: float) -> float:
        """Calculate number of ticks between two prices."""
        return abs(price_a - price_b) / self.tick_size

    def micro_dollar_risk(self, entry: float, stop: float, contracts: int = 1) -> float:
        """Calculate dollar risk for micro contracts."""
        ticks = self.ticks_between(entry, stop)
        return ticks * self.tick_value * contracts

    def full_dollar_risk(self, entry: float, stop: float, contracts: int = 1) -> float:
        """Calculate dollar risk for full contracts."""
        ticks = self.ticks_between(entry, stop)
        return ticks * self.full_tick_value * contracts


# Pre-defined contract specifications for supported instruments
CONTRACTS: dict[str, ContractSpec] = {
    "ES": ContractSpec(
        symbol="ES",
        micro_symbol="MES",
        name="E-mini S&P 500",
        exchange="CME",
        tick_size=0.25,
        tick_value=1.25,  # MES: $1.25/tick
        full_tick_value=12.50,  # ES: $12.50/tick
        multiplier=10,
        rth_open="09:30",
        rth_close="16:00",
        eth_open="18:00",
        eth_close="17:00",
    ),
    "NQ": ContractSpec(
        symbol="NQ",
        micro_symbol="MNQ",
        name="E-mini Nasdaq 100",
        exchange="CME",
        tick_size=0.25,
        tick_value=0.50,  # MNQ: $0.50/tick
        full_tick_value=5.00,  # NQ: $5.00/tick
        multiplier=10,
        rth_open="09:30",
        rth_close="16:00",
        eth_open="18:00",
        eth_close="17:00",
    ),
    "YM": ContractSpec(
        symbol="YM",
        micro_symbol="MYM",
        name="E-mini Dow Jones",
        exchange="CBOT",
        tick_size=1.0,
        tick_value=0.50,  # MYM: $0.50/tick
        full_tick_value=5.00,  # YM: $5.00/tick
        multiplier=10,
        rth_open="09:30",
        rth_close="16:00",
        eth_open="18:00",
        eth_close="17:00",
    ),
    "RTY": ContractSpec(
        symbol="RTY",
        micro_symbol="M2K",
        name="E-mini Russell 2000",
        exchange="CME",
        tick_size=0.10,
        tick_value=0.50,  # M2K: $0.50/tick
        full_tick_value=5.00,  # RTY: $5.00/tick
        multiplier=10,
        rth_open="09:30",
        rth_close="16:00",
        eth_open="18:00",
        eth_close="17:00",
    ),
    "CL": ContractSpec(
        symbol="CL",
        micro_symbol="MCL",
        name="Crude Oil",
        exchange="NYMEX",
        tick_size=0.01,
        tick_value=1.00,  # MCL: $1.00/tick
        full_tick_value=10.00,  # CL: $10.00/tick
        multiplier=10,
        rth_open="09:00",
        rth_close="14:30",
        eth_open="18:00",
        eth_close="17:00",
    ),
    "GC": ContractSpec(
        symbol="GC",
        micro_symbol="MGC",
        name="Gold",
        exchange="COMEX",
        tick_size=0.10,
        tick_value=1.00,  # MGC: $1.00/tick
        full_tick_value=10.00,  # GC: $10.00/tick
        multiplier=10,
        rth_open="08:20",
        rth_close="13:30",
        eth_open="18:00",
        eth_close="17:00",
    ),
    "SI": ContractSpec(
        symbol="SI",
        micro_symbol="SIL",
        name="Silver",
        exchange="COMEX",
        tick_size=0.005,
        tick_value=2.50,  # SIL: $2.50/tick (1000 oz micro)
        full_tick_value=25.00,  # SI: $25.00/tick
        multiplier=10,
        rth_open="08:25",
        rth_close="13:25",
        eth_open="18:00",
        eth_close="17:00",
    ),
    "ZB": ContractSpec(
        symbol="ZB",
        micro_symbol="ZB",  # No micro for ZB, trade full
        name="30-Year Treasury Bond",
        exchange="CBOT",
        tick_size=0.03125,  # 1/32nd
        tick_value=31.25,
        full_tick_value=31.25,
        multiplier=1,
        rth_open="08:20",
        rth_close="15:00",
        eth_open="18:00",
        eth_close="17:00",
    ),
    "ZN": ContractSpec(
        symbol="ZN",
        micro_symbol="ZN",  # No micro for ZN, trade full
        name="10-Year Treasury Note",
        exchange="CBOT",
        tick_size=0.015625,  # 1/64th
        tick_value=15.625,
        full_tick_value=15.625,
        multiplier=1,
        rth_open="08:20",
        rth_close="15:00",
        eth_open="18:00",
        eth_close="17:00",
    ),
}


def get_contract(symbol: str) -> ContractSpec:
    """Get contract spec by symbol. Raises KeyError if not found."""
    symbol = symbol.upper()
    if symbol not in CONTRACTS:
        raise KeyError(
            f"Unknown contract symbol: {symbol}. Supported: {', '.join(sorted(CONTRACTS.keys()))}"
        )
    return CONTRACTS[symbol]

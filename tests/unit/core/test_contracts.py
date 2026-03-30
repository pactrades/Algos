"""Tests for the ContractSpec model and contract registry."""

import pytest

from algos.core.contracts import CONTRACTS, get_contract


class TestContractSpec:
    """Test ContractSpec data and calculations."""

    def test_es_contract_exists(self) -> None:
        es = get_contract("ES")
        assert es.symbol == "ES"
        assert es.micro_symbol == "MES"
        assert es.tick_size == 0.25
        assert es.tick_value == 1.25
        assert es.full_tick_value == 12.50

    def test_nq_contract_exists(self) -> None:
        nq = get_contract("NQ")
        assert nq.symbol == "NQ"
        assert nq.micro_symbol == "MNQ"
        assert nq.tick_size == 0.25
        assert nq.tick_value == 0.50

    def test_cl_contract_exists(self) -> None:
        cl = get_contract("CL")
        assert cl.symbol == "CL"
        assert cl.micro_symbol == "MCL"
        assert cl.tick_size == 0.01
        assert cl.tick_value == 1.00

    def test_gc_contract_exists(self) -> None:
        gc = get_contract("GC")
        assert gc.symbol == "GC"
        assert gc.micro_symbol == "MGC"
        assert gc.tick_size == 0.10
        assert gc.tick_value == 1.00

    def test_all_supported_contracts_present(self) -> None:
        expected = {"ES", "NQ", "YM", "RTY", "CL", "GC", "SI", "ZB", "ZN"}
        assert set(CONTRACTS.keys()) == expected

    def test_case_insensitive_lookup(self) -> None:
        assert get_contract("es").symbol == "ES"
        assert get_contract("Es").symbol == "ES"

    def test_unknown_contract_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown contract symbol"):
            get_contract("FAKE")


class TestTickCalculations:
    """Test tick-based calculations."""

    def test_ticks_between_es(self) -> None:
        es = get_contract("ES")
        # 10 points / 0.25 tick size = 40 ticks
        assert es.ticks_between(5000.0, 5010.0) == pytest.approx(40.0)

    def test_ticks_between_is_absolute(self) -> None:
        es = get_contract("ES")
        assert es.ticks_between(5010.0, 5000.0) == pytest.approx(40.0)

    def test_micro_dollar_risk_es(self) -> None:
        es = get_contract("ES")
        # 10 points = 40 ticks * $1.25/tick = $50 per micro contract
        risk = es.micro_dollar_risk(5000.0, 5010.0, contracts=1)
        assert risk == pytest.approx(50.0)

    def test_micro_dollar_risk_multiple_contracts(self) -> None:
        es = get_contract("ES")
        risk = es.micro_dollar_risk(5000.0, 5010.0, contracts=3)
        assert risk == pytest.approx(150.0)

    def test_full_dollar_risk_es(self) -> None:
        es = get_contract("ES")
        # 10 points = 40 ticks * $12.50/tick = $500 per full contract
        risk = es.full_dollar_risk(5000.0, 5010.0, contracts=1)
        assert risk == pytest.approx(500.0)

    def test_micro_dollar_risk_cl(self) -> None:
        cl = get_contract("CL")
        # $1.00 move = 100 ticks * $1.00/tick = $100 per micro
        risk = cl.micro_dollar_risk(75.00, 76.00, contracts=1)
        assert risk == pytest.approx(100.0)

    def test_micro_dollar_risk_gc(self) -> None:
        gc = get_contract("GC")
        # $10 move = 100 ticks * $1.00/tick = $100 per micro
        risk = gc.micro_dollar_risk(2050.0, 2060.0, contracts=1)
        assert risk == pytest.approx(100.0)

"""Tests for YAML configuration loader."""

from pathlib import Path

import pytest

from algos.core.config import (
    load_app_config,
    load_prop_firm_profile,
    load_strategy_config,
    load_yaml,
)

CONFIG_DIR = Path(__file__).parents[3] / "config"


class TestLoadYaml:
    """Test raw YAML loading."""

    def test_load_existing_yaml(self) -> None:
        data = load_yaml(CONFIG_DIR / "default.yaml")
        assert isinstance(data, dict)
        assert "global_max_positions" in data

    def test_load_nonexistent_yaml_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_yaml(Path("/nonexistent/file.yaml"))


class TestPropFirmProfile:
    """Test prop firm profile loading."""

    def test_load_apex_profile(self) -> None:
        profile = load_prop_firm_profile(CONFIG_DIR / "prop_firms" / "apex.yaml")
        assert profile.name == "Apex Trader Funding"
        assert profile.account_size == 50000
        assert profile.trailing_drawdown == 2500
        assert profile.profit_target == 3000

    def test_load_topstep_profile(self) -> None:
        profile = load_prop_firm_profile(CONFIG_DIR / "prop_firms" / "topstep.yaml")
        assert profile.name == "TopStep"
        assert profile.daily_loss_limit == 1000

    def test_all_prop_firm_profiles_load(self) -> None:
        for yaml_file in (CONFIG_DIR / "prop_firms").glob("*.yaml"):
            profile = load_prop_firm_profile(yaml_file)
            assert profile.account_size > 0
            assert profile.trailing_drawdown > 0

    def test_invalid_profile_negative_drawdown(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text(
            "name: Bad\naccount_size: 50000\ntrailing_drawdown: -100\n"
            "daily_loss_limit: 1000\nprofit_target: 3000\nmax_contracts: 5\n"
        )
        with pytest.raises(ValueError):
            load_prop_firm_profile(bad_yaml)


class TestStrategyConfig:
    """Test strategy configuration loading."""

    def test_load_temporal_edge_config(self) -> None:
        config = load_strategy_config(CONFIG_DIR / "strategies" / "temporal_edge.yaml")
        assert config.name == "temporal_edge"
        assert config.enabled is True
        assert "ES" in config.instruments
        assert config.risk_per_trade_pct == 1.0

    def test_all_strategy_configs_load(self) -> None:
        expected_names = {
            "temporal_edge",
            "intermarket_flow",
            "vol_regime",
            "micro_structure",
            "calendar_alpha",
        }
        loaded_names = set()
        for yaml_file in (CONFIG_DIR / "strategies").glob("*.yaml"):
            config = load_strategy_config(yaml_file)
            loaded_names.add(config.name)
        assert loaded_names == expected_names


class TestAppConfig:
    """Test full application config assembly."""

    def test_load_app_config_with_apex(self) -> None:
        config = load_app_config(CONFIG_DIR, prop_firm_name="apex")
        assert config.prop_firm.name == "Apex Trader Funding"
        assert len(config.strategies) == 5
        assert "temporal_edge" in config.strategies
        assert config.global_max_positions == 5

    def test_load_app_config_missing_firm_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_app_config(CONFIG_DIR, prop_firm_name="nonexistent")

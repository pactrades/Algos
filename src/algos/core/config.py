"""YAML configuration loader with validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PropFirmProfile(BaseModel):
    """Configuration for a prop firm's rules and constraints."""

    name: str
    trailing_drawdown: float = Field(gt=0, description="Max trailing drawdown in dollars")
    daily_loss_limit: float = Field(gt=0, description="Max daily loss in dollars")
    profit_target: float = Field(gt=0, description="Evaluation profit target in dollars")
    max_contracts: int = Field(gt=0, description="Maximum concurrent contracts")
    account_size: float = Field(gt=0, description="Account size in dollars")


class StrategyConfig(BaseModel):
    """Configuration for a single strategy."""

    name: str
    enabled: bool = True
    instruments: list[str] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=list)
    max_concurrent_positions: int = Field(default=3, gt=0)
    risk_per_trade_pct: float = Field(default=1.0, gt=0, le=5.0)
    parameters: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    """Top-level application configuration."""

    prop_firm: PropFirmProfile
    strategies: dict[str, StrategyConfig] = Field(default_factory=dict)
    global_max_positions: int = Field(default=5, gt=0)
    global_risk_pct: float = Field(default=2.0, gt=0, le=10.0)
    output_handlers: list[str] = Field(default_factory=lambda: ["console"])


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict at top level of {path}, got {type(data).__name__}")
    return data


def load_prop_firm_profile(path: Path) -> PropFirmProfile:
    """Load a prop firm profile from a YAML file."""
    data = load_yaml(path)
    return PropFirmProfile(**data)


def load_strategy_config(path: Path) -> StrategyConfig:
    """Load a strategy configuration from a YAML file."""
    data = load_yaml(path)
    return StrategyConfig(**data)


def load_app_config(config_dir: Path, prop_firm_name: str = "apex") -> AppConfig:
    """Load the full application config from a config directory.

    Loads the prop firm profile and all strategy configs, then assembles
    them into an AppConfig.
    """
    # Load prop firm profile
    firm_path = config_dir / "prop_firms" / f"{prop_firm_name}.yaml"
    if not firm_path.exists():
        raise FileNotFoundError(f"Prop firm profile not found: {firm_path}")
    prop_firm = load_prop_firm_profile(firm_path)

    # Load all strategy configs
    strategies: dict[str, StrategyConfig] = {}
    strategies_dir = config_dir / "strategies"
    if strategies_dir.exists():
        for yaml_file in sorted(strategies_dir.glob("*.yaml")):
            strat = load_strategy_config(yaml_file)
            strategies[strat.name] = strat

    # Load default config for global settings
    default_path = config_dir / "default.yaml"
    defaults = load_yaml(default_path) if default_path.exists() else {}

    return AppConfig(
        prop_firm=prop_firm,
        strategies=strategies,
        global_max_positions=defaults.get("global_max_positions", 5),
        global_risk_pct=defaults.get("global_risk_pct", 2.0),
        output_handlers=defaults.get("output_handlers", ["console"]),
    )

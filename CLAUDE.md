# CLAUDE.md — Development Guidelines

## Project Overview

**Algos** is a futures trading signal generation platform. It discovers novel patterns from historical data and deploys them as validated trading algorithms. Signals are executed on micro contracts to pass retail prop firm challenges while hedging the user's manual NQ trading.

## Architecture

```
src/algos/
├── core/         # Signal model, contracts, risk, session times, config
├── research/     # Statistics, validation engine, pattern scanners
├── strategies/   # 5 algorithm implementations (one per category)
├── output/       # Signal delivery: console, webhook, Telegram
└── runner.py     # Orchestrator
```

### Key Concepts
- **Signal**: The universal output — every strategy produces `Signal` objects
- **ContractSpec**: Tick size, tick value, trading hours per futures instrument
- **Pattern Scanner**: Systematic search for statistical edges in historical data
- **Validation Engine**: Walk-forward, OOS, Monte Carlo — every pattern must pass before deployment
- **Prop Firm Profile**: Configurable rules (drawdown, daily loss, contract limits)

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest                           # All tests
pytest tests/unit/ -v            # Unit tests only
pytest tests/integration/ -v     # Integration tests
pytest tests/validation/ -v      # Validation tests
pytest -m "not slow"             # Skip slow tests

# Linting & type checking
ruff check src/ tests/           # Lint
ruff format src/ tests/          # Format
mypy src/                        # Type check

# Coverage
pytest --cov=src/algos --cov-report=term-missing

# Scripts
python scripts/download_data.py  # Download parquet files from Google Drive
python scripts/run_signals.py    # Run strategies and emit signals
python scripts/research.py       # Run pattern discovery pipeline
python scripts/validate.py       # Validate a discovered pattern
```

## Code Style

- Python 3.11+, strict typing
- Pydantic for data models
- pytest for all testing (TDD: write tests first, verify red, then implement)
- ruff for linting/formatting (line length: 100)
- mypy strict mode
- No pre-existing/textbook trading strategies — all patterns discovered from data

## Key Constraints

- **NQ Blackout**: No automated NQ/MNQ signals between 9:30 AM - 12:00 PM ET
- **Micro Contracts Only**: All position sizing in micro contracts (MES, MNQ, MCL, etc.)
- **Data Source**: Full contract data for analysis, micro contract sizing for execution
- **Prop Firm Compliance**: Every signal must pass risk management checks before emission
- **Pattern Validation**: p < 0.05, OOS profitable, walk-forward >60% windows profitable, Monte Carlo P(ruin) < 5%

## Config

- `config/prop_firms/` — Prop firm profiles (Apex, TopStep, Earn2Trade, Bulenox)
- `config/strategies/` — Per-strategy parameters
- `config/contracts.yaml` — Futures contract specifications
- `config/default.yaml` — Global defaults

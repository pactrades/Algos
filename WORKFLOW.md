# WORKFLOW.md — TDD Workflow & Development Process

## TDD Cycle (Red -> Green -> Refactor)

Every feature follows this strict cycle:

### 1. RED — Write Failing Tests First
- Write tests that define the expected behavior
- Run tests and **verify they FAIL** (red)
- Tests should fail for the right reason (e.g., `ImportError`, `AttributeError`, not syntax errors)

### 2. GREEN — Implement Minimum Code
- Write the minimum code to make all tests pass
- Run tests and **verify they PASS** (green)
- No gold-plating — only what the tests require

### 3. REFACTOR — Clean Up
- Improve code quality without changing behavior
- Run tests again to confirm they still pass
- Lint (`ruff check`) and type-check (`mypy`)

## PR & Review Process

### Per-Feature Flow
1. **Branch**: Work on the designated feature branch
2. **TDD**: Follow Red -> Green -> Refactor cycle
3. **Commit**: Clear, descriptive commit messages
4. **Push**: Push to remote branch
5. **CI**: Verify all CI checks pass (lint, type-check, tests)
6. **Audit**: Zero-context review — read the diff as if seeing it for the first time
7. **Quiz**: Knowledge transfer — quiz on how the code works

### Commit Message Format
```
<type>(<scope>): <description>

Types: feat, fix, refactor, test, docs, chore
Scopes: core, research, strategies, output, config, ci
```

Examples:
- `feat(core): add Signal pydantic model with validation`
- `test(core): add unit tests for ContractSpec tick calculations`
- `docs: create CLAUDE.md and WORKFLOW.md`

## Pattern Discovery Workflow

### Phase 1: Scan
- Run pattern scanner against historical data
- Log all candidate patterns with basic statistics

### Phase 2: Filter
- Minimum 100 trade occurrences
- p-value < 0.05 for edge over random
- Effect size (Cohen's d) > 0.2

### Phase 3: Validate
Run the full validation pipeline:

1. **In-Sample / Out-of-Sample Split** (70/30)
   - Pattern must be profitable in BOTH periods
   - OOS Sharpe ratio > 0.5

2. **Walk-Forward Optimization**
   - Rolling 6-month IS, 2-month OOS windows
   - Profitable in >60% of OOS windows
   - No single OOS window with >20% drawdown

3. **Monte Carlo Simulation** (10,000 iterations)
   - 95th percentile max DD within prop firm limits
   - Median annual return positive
   - Probability of ruin < 5%

4. **Robustness Checks**
   - Parameter sensitivity: edge holds at +/-20% variation
   - Multi-instrument: works on >=2 instruments
   - Regime analysis: identify failure regimes

### Phase 4: Deploy
- Implement as strategy class
- Write config file with discovered parameters
- Full unit test coverage
- Integration test with runner

## Testing Strategy

### Test Pyramid
```
        /  Validation  \        <- Backtest & statistical validation
       / Integration    \       <- Full pipeline tests
      /    Unit Tests    \      <- Individual components
```

### Test Categories
- **Unit** (`tests/unit/`): Individual functions, classes, edge cases
- **Integration** (`tests/integration/`): Multi-component pipelines
- **Validation** (`tests/validation/`): Statistical validation results

### Test Data
- Synthetic OHLCV data with known patterns (in `conftest.py`)
- Historical data for backtests (in `data/`, gitignored)
- Never test against live data in CI

## Quiz Protocol

After each feature is complete and merged, a quiz covers:
1. **What** — What does the code do?
2. **Why** — Why was it designed this way?
3. **How** — How do the key functions work?
4. **Edge Cases** — What happens in unusual scenarios?
5. **Integration** — How does it connect to the rest of the system?

This ensures the user fully understands every component of the system.

## Epic Execution Order

```
Epic 1: Scaffolding & Core Infrastructure  <- CURRENT
Epic 2: Data Pipeline
Epic 3: Statistical Toolkit     }
Epic 4: Validation Engine       } <- parallel
Epic 5: Pattern Scanners
Epic 6: Risk Management  }
Epic 7: Strategy Framework} <- parallel
Epic 8: Signal Output     }
Epics 9-13: Strategy Implementations (parallel)
Epic 14: Integration & Final Validation
```

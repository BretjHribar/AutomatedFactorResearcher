# Automated Factor Researcher

An automated equity factor alpha research, backtesting, combination, and portfolio optimization platform comparable to WorldQuant BRAIN/WebSim.

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Linux/Mac

# Install
pip install -e ".[dev]"

# Generate fixture data
python scripts/generate_fixtures.py

# Run tests
pytest

# Start API server
uvicorn src.api.main:app --reload
```

## Architecture

- **Core Engine**: Simulation loop matching BRAIN/WebSim metrics exactly
- **Operators Library**: Full WebSim-compatible cross-sectional, time-series, and element-wise operators
- **Expression Parser**: Parse `"-rank(delta(close, 5))"` into executable alpha functions
- **Synthetic Data**: Zero external dependencies for development — built-in data generator with embedded signals
- **DataContext Abstraction**: Factor code never references a specific vendor

## Project Status

Phase 1 (Core Engine) — In Progress

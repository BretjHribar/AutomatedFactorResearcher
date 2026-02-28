# Automated Factor Researcher

An automated equity factor alpha research, backtesting, combination, and portfolio optimization platform comparable to [WorldQuant BRAIN / WebSim](https://platform.worldquantbrain.com/). Designed for systematic discovery and evaluation of cross-sectional equity alpha factors using the same expression language, operator library, simulation methodology, and neutralization framework as WorldQuant.

## Replication Accuracy

This platform closely replicates WorldQuant BRAIN simulation results:

| Metric | WorldQuant BRAIN | This Platform | Gap |
|--------|-----------------|---------------|-----|
| **Sharpe Ratio** | 2.10 | 2.04 | 0.06 (3%) |
| **Universe** | TOP3000 US | TOP3000 US | — |
| **Neutralization** | GICS Subindustry | GICS Subindustry | — |
| **Classification** | GICS (8-digit) | GICS (8-digit) | — |
| **Period** | 2019–2023 | 2019–2023 | — |

> Tested with: `rank(ts_regression(sales, invested_capital, 252, lag=126, rettype=2)) * rank(ts_zscore(divide(return_equity, invested_capital), 252)) * rank(group_rank(divide(liabilities_curr, assets), subindustry))`

---

## Table of Contents

- [Architecture](#architecture)
- [Data Pipeline](#data-pipeline)
- [Operator Library](#operator-library)
- [Expression Engine](#expression-engine)
- [Simulation Engine](#simulation-engine)
- [Alpha Discovery](#alpha-discovery)
- [API Server](#api-server)
- [Web GUI](#web-gui)
- [Quick Start](#quick-start)
- [Data Fields](#data-fields)
- [Project Structure](#project-structure)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Web GUI (Alpha Library)                   │
│              gui/index.html · app.js · portfolio.js              │
└─────────────────────────────┬────────────────────────────────────┘
                              │ HTTP / WebSocket
┌─────────────────────────────▼────────────────────────────────────┐
│                    FastAPI Server (src/api/server.py)             │
│    POST /simulate  ·  POST /gp/start  ·  GET /alphas  ·  WS     │
└──────┬──────────────────┬───────────────────────┬────────────────┘
       │                  │                       │
┌──────▼──────┐   ┌───────▼───────┐   ┌──────────▼──────────┐
│ Expression  │   │  Simulation   │   │   Alpha Discovery   │
│   Engine    │   │    Engine     │   │                     │
│             │   │               │   │  • GP Engine (DEAP) │
│ Lexer →     │   │ • Vectorized  │   │  • LLM Agent        │
│ Parser →    │   │ • Delay/Decay │   │    (Gemini)         │
│ AST →       │   │ • Neutralize  │   │  • Evaluation       │
│ Evaluator   │   │ • PnL Calc    │   │    Pipeline         │
└──────┬──────┘   └───────┬───────┘   └──────────┬──────────┘
       │                  │                       │
┌──────▼──────────────────▼───────────────────────▼────────────────┐
│                      Data Layer                                  │
│                                                                  │
│  InMemoryDataContext  ·  FMP Loader  ·  Alpha Database (SQLite)  │
│  121 Parquet matrices · GICS classifications · 5 universe tiers  │
└──────────────────────────────────────────────────────────────────┘
```

### Core Components

| Module | Path | Description |
|--------|------|-------------|
| **Expression Engine** | `src/operators/fastexpression.py` | Compiles WQ BRAIN expression strings into vectorized DataFrame operations |
| **Operator Library** | `src/operators/vectorized.py` | 80+ operators: time-series, cross-sectional, element-wise, group |
| **Simulation** | `src/simulation/vectorized_sim.py` | Full vectorized backtester matching BRAIN metrics |
| **GP Engine** | `src/agent/gp_engine.py` | Genetic programming alpha discovery using DEAP |
| **LLM Agent** | `src/agent/research_agent.py` | Gemini-powered alpha generation and iteration |
| **API Server** | `src/api/server.py` | FastAPI REST + WebSocket server |
| **Data Loader** | `src/data/fmp_loader.py` | FMP Parquet matrix loader with universe filtering |
| **Alpha DB** | `src/data/alpha_database.py` | SQLite storage for discovered alphas and evaluations |
| **Portfolio** | `src/portfolio/optimizer.py` | Multi-alpha combination and portfolio optimization |

---

## Data Pipeline

### Data Sources

All market data is sourced from [Financial Modeling Prep (FMP)](https://financialmodelingprep.com/) and stored as Parquet matrices in `data/fmp_cache/matrices/`.

| Category | Data |
|----------|------|
| **Prices** | OHLCV, VWAP (7,835 tickers including delisted) |
| **Fundamentals** | Income statement, balance sheet, cash flow, key metrics (2,512 tickers, quarterly) |
| **Derived Ratios** | 33 computed fields: ROE, ROA, margins, yields, per-share metrics |
| **Volatility** | Historical & Parkinson volatility at 10/20/30/60/90/120/150/180-day windows |
| **Universes** | TOP200, TOP500, TOP1000, TOP2000, TOP3000 (ADV20-ranked, rebalanced every 20 days) |

### Matrix Format

Every data field is stored as a **Parquet DataFrame** with shape `(dates × tickers)`:
- **Index**: DatetimeIndex (2016-01-04 to present, ~2,552 trading days)
- **Columns**: Ticker symbols (up to 7,835 unique)
- **Values**: Float64, NaN for missing data

### Survivorship Bias Fix

The platform downloads **delisted companies** from FMP to prevent survivorship bias:
- 5,415 delisted stocks added to the raw price matrices
- Historical prices preserved for stocks that were acquired, went bankrupt, or delisted
- Universes rebuilt to include these stocks in the periods when they were actively traded
- `expand_universe.py` — script for downloading delisted data

### GICS Classification System

Stock classifications use the **Global Industry Classification Standard (GICS)**, matching WorldQuant BRAIN exactly:

| Level | Code Length | Example | Count |
|-------|------------|---------|-------|
| Sector | 2-digit | `45` (Information Technology) | 11 |
| Industry Group | 4-digit | `4520` (Technology Hardware) | ~25 |
| Industry | 6-digit | `452030` (Electronic Equipment) | ~69 |
| **Sub-Industry** | **8-digit** | `45203010` (Electronic Equipment & Instruments) | **106** |

- Built by `build_gics_classifications.py` which maps FMP sector/industry names to official GICS codes
- Stored in `data/fmp_cache/classifications.json`
- Used for `group_rank()`, `group_neutralize()`, and simulation neutralization
- 100% coverage across all 2,512 fundamental-data tickers

### Building the Data

```bash
# Step 1: Download raw data from FMP (requires API key)
export FMP_API_KEY="your_key_here"
python -m src.data.bulk_download

# Step 2: Build GICS classifications
python build_gics_classifications.py

# Step 3: Compute derived financial ratios
python build_derived_fields.py

# Step 4: (Optional) Add delisted stocks to fix survivorship bias
python expand_universe.py
```

---

## Operator Library

The operator library (`src/operators/vectorized.py`) implements 80+ operators matching WorldQuant BRAIN's fastexpression semantics. All operators work on full DataFrames — no per-date loops (except `ts_regression`).

### Time-Series Operators (per-instrument over time)

| Operator | Signature | Description |
|----------|-----------|-------------|
| `ts_sum` | `ts_sum(x, d)` | Rolling sum over past `d` days |
| `ts_mean` / `sma` | `ts_mean(x, d)` | Simple moving average |
| `ts_rank` | `ts_rank(x, d)` | Percentile rank within past `d` days |
| `ts_min` / `ts_max` | `ts_min(x, d)` | Rolling min/max |
| `ts_delta` / `delta` | `ts_delta(x, d)` | `x[t] - x[t-d]` |
| `ts_std_dev` / `stddev` | `ts_std_dev(x, d)` | Rolling standard deviation |
| `ts_corr` / `correlation` | `ts_corr(x, y, d)` | Rolling correlation |
| `ts_cov` / `covariance` | `ts_cov(x, y, d)` | Rolling covariance |
| `ts_zscore` | `ts_zscore(x, d)` | `(x - ma) / std` (look-back only) |
| `ts_regression` | `ts_regression(y, x, d, lag, rettype)` | OLS regression (returns residual/slope/intercept/fitted) |
| `ts_skewness` | `ts_skewness(x, d)` | Rolling skewness |
| `ts_kurtosis` | `ts_kurtosis(x, d)` | Rolling kurtosis |
| `ts_argmax` / `ts_argmin` | `ts_argmax(x, d)` | Index of max/min in window |
| `decay_linear` | `decay_linear(x, d)` | Linear decay weighted MA |
| `decay_exp` | `decay_exp(x, α)` | Exponential decay (EWM) |
| `delay` | `delay(x, d)` | Shift by `d` days |
| `ts_entropy` | `ts_entropy(x, d)` | Rolling entropy approximation |

### Cross-Sectional Operators (across instruments)

| Operator | Signature | Description |
|----------|-----------|-------------|
| `rank` | `rank(x)` | Percentile rank across all instruments `[0, 1]` |
| `zscore` | `zscore(x)` | `(x - mean) / std` across instruments |
| `scale` | `scale(x, k)` | Scale so `sum(abs(x)) = k` |
| `normalize` | `normalize(x)` | Normalize to `[-1, 1]` |
| `winsorize` | `winsorize(x)` | Clip to percentile limits |
| `truncate` | `truncate(x, w)` | Clip absolute values to `w` |
| `quantile` | `quantile(x)` | Quantile transformation |

### Group Operators (within GICS groups)

| Operator | Signature | Description |
|----------|-----------|-------------|
| `group_rank` | `group_rank(x, group)` | Percentile rank within each group |
| `group_neutralize` | `group_neutralize(x, group)` | Demean within each group |
| `group_zscore` | `group_zscore(x, group)` | Z-score within each group |
| `group_scale` | `group_scale(x, group)` | Scale within each group |
| `group_mean` | `group_mean(x, w, group)` | Weighted mean within each group |

### Element-Wise Operators

`add`, `subtract`, `multiply`, `divide`, `abs`, `sign`, `signed_power`, `log`, `sqrt`, `square`, `inverse`, `power`, `max`, `min`, `if_else`, `pasteurize`, `s_log_1p`, `hump`, `tail`, `bucket`

---

## Expression Engine

The expression engine (`src/operators/fastexpression.py`) compiles WorldQuant BRAIN expression strings into callable Python functions:

```python
from src.operators.fastexpression import FastExpressionEngine

engine = FastExpressionEngine(data_fields={'close': df_close, 'volume': df_volume, ...})
engine.add_group('subindustry', gics_subindustry_series)

# Evaluate any BRAIN expression
alpha = engine.evaluate("rank(ts_delta(divide(close, volume), 120))")
alpha = engine.evaluate("-rank(delta(close, 5))")
alpha = engine.evaluate("rank(ts_regression(revenue, assets, 220, lag=110, rettype=2))")
alpha = engine.evaluate("group_neutralize(rank(-delta(close, 5)), subindustry)")
```

### Supported Syntax

- **Arithmetic**: `+`, `-`, `*`, `/`, `^`
- **Comparison**: `<`, `<=`, `>`, `>=`, `==`, `!=`
- **Logical**: `||`, `&&`, `!`
- **Ternary**: `cond ? true_val : false_val`
- **Functions**: All 80+ registered operators
- **Keyword args**: `ts_regression(y, x, 252, lag=126, rettype=2)`
- **Group levels**: `subindustry`, `industry`, `sector`

---

## Simulation Engine

The simulation engine (`src/simulation/vectorized_sim.py`) is a fully vectorized backtester that matches BRAIN/WebSim metrics:

### Simulation Pipeline

1. **Clean**: Replace inf with NaN, forward-fill gaps
2. **Delay**: Apply trade execution delay (default: 1 day)
3. **Decay**: Optional exponential or linear decay
4. **Universe Filter**: Mask to TOP200/500/1000/2000/3000
5. **Neutralize**: Demean within GICS groups (sector/industry/subindustry)
6. **Normalize**: Scale to unit sum (dollar-neutral)
7. **Clip**: Enforce max stock weight (default: 1%)
8. **PnL**: Compute daily returns, Sharpe, drawdown, turnover

### Metrics Computed

| Metric | Description |
|--------|-------------|
| **Sharpe Ratio** | Annualized (×√252) |
| **Fitness** | Sharpe × √(abs(returns)) × sign(returns) |
| **Annual Return** | Annualized cumulative return |
| **Turnover** | Average daily position change |
| **Max Drawdown** | Maximum peak-to-trough decline |
| **Margin** | Average daily PnL per dollar traded (bps) |
| **Total PnL** | Cumulative dollar PnL |

### Out-of-Sample Testing

The OOS module (`src/simulation/oos.py`) provides:
- Walk-forward validation with configurable IS/OOS split
- Expanding and rolling window modes
- Bootstrap confidence intervals

---

## Alpha Discovery

### Genetic Programming (GP)

The GP engine (`src/agent/gp_engine.py`) uses [DEAP](https://github.com/deap/deap) to evolve alpha expressions:

```bash
python run_gp_top1000.py
```

Features:
- Tree-based GP with crossover, mutation, and selection
- Curated feature set to avoid combinatorial explosion
- Fitness = Sharpe × √|returns| with turnover and drawdown constraints
- Multi-threaded evaluation (cross-thread SQLite access)
- Automatic storage of passing alphas to the database

### LLM Agent (Gemini)

The research agent (`src/agent/research_agent.py`) uses Google Gemini to generate alpha ideas:

- Iterates on alpha expressions based on feedback from previous simulations
- Uses available data field catalog and operator documentation
- Stores passing alphas with reasoning/provenance

### Evaluation Pipeline

The pipeline (`src/evaluation/pipeline.py`) orchestrates:
1. Expression parsing and evaluation
2. In-sample simulation with neutralization
3. Quality checks (|Sharpe| > 0.3, turnover > 0.001)
4. Database storage with full metrics and metadata

---

## API Server

The FastAPI server (`src/api/server.py`) provides:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/simulate` | POST | Simulate an alpha expression |
| `/fields` | GET | List available data fields |
| `/alphas` | GET | List stored alphas |
| `/alphas/{id}` | GET | Get alpha details |
| `/gp/start` | POST | Start GP evolution |
| `/gp/stop` | POST | Stop GP evolution |
| `/ws` | WebSocket | Real-time GP progress updates |

```bash
uvicorn src.api.server:app --reload --port 8000
```

---

## Web GUI

The Alpha Library GUI (`gui/`) provides a rich interface for:

- **Alpha Explorer**: Browse, filter, and sort discovered alphas
- **Live Simulation**: Enter expressions and see results in real-time
- **GP Dashboard**: Monitor genetic programming runs with WebSocket updates
- **Portfolio View**: Multi-alpha combination and portfolio construction

Access at `http://localhost:8000` when the API server is running.

---

## Quick Start

### Prerequisites

- Python 3.12+
- FMP API key (set as `FMP_API_KEY` environment variable)

### Installation

```bash
# Clone and enter the subproject
cd factor-alpha-platform

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Linux/Mac

# Install with dev dependencies
pip install -e ".[dev]"
```

### Run with Existing Data

```bash
# Start the API server
uvicorn src.api.server:app --reload --port 8000

# Run GP alpha discovery
python run_gp_top1000.py

# Simulate a specific alpha expression
python test_wq_alpha.py
```

### Run Tests

```bash
pytest
pytest -m "not slow"   # skip slow tests
pytest --cov=src       # with coverage
```

---

## Data Fields

### 121 Available Parquet Matrices

<details>
<summary><strong>Price Data</strong> (7,835 tickers including delisted)</summary>

`close`, `open`, `high`, `low`, `volume`, `vwap`, `returns`, `log_returns`, `adv20`, `adv60`, `dollars_traded`

</details>

<details>
<summary><strong>Fundamental Data</strong> (2,512 tickers, quarterly-filled daily)</summary>

**Income Statement**: `revenue`, `sales`, `gross_profit`, `operating_income`, `ebit`, `ebitda`, `income`, `net_income`, `eps`, `eps_diluted`, `cost_of_revenue`, `cogs`, `operating_expense`, `sga_expense`, `rd_expense`, `interest_expense`, `income_tax`, `depreciation`, `depre_amort`

**Balance Sheet**: `assets`, `total_assets`, `assets_curr`, `cash`, `receivables`, `inventory`, `goodwill`, `intangibles`, `ppe_net`, `liabilities`, `total_liabilities`, `liabilities_curr`, `payables`, `debt`, `total_debt`, `debt_lt`, `equity`, `total_equity`, `retained_earnings`, `working_capital`

**Cash Flow**: `cashflow`, `cashflow_op`, `cashflow_invst`, `free_cashflow`, `capex`, `stock_repurchase`

**Key Metrics**: `market_cap`, `cap`, `enterprise_value`, `invested_capital`, `current_ratio`, `inventory_turnover`, `net_debt`

</details>

<details>
<summary><strong>Derived Ratios</strong> (computed by build_derived_fields.py)</summary>

**Profitability**: `roe`, `return_equity`, `roa`, `return_assets`, `gross_margin`, `operating_margin`, `net_margin`, `ebitda_margin`, `asset_turnover`

**Valuation**: `pe_ratio`, `pb_ratio`, `ev_to_ebitda`, `ev_to_revenue`, `ev_to_fcf`, `earnings_yield`, `free_cashflow_yield`, `book_to_market`

**Leverage**: `debt_to_equity`, `debt_to_assets`, `cash_ratio`, `quick_ratio`, `interest_coverage`

**Per-Share**: `bookvalue_ps`, `fcf_per_share`, `revenue_per_share`, `sales_ps`, `tangible_book_per_share`, `sharesout`

**Efficiency**: `capex_to_revenue`, `rd_to_revenue`, `sga_to_revenue`, `cash_conversion_ratio`

</details>

<details>
<summary><strong>Volatility</strong> (computed from returns)</summary>

`historical_volatility_{10,20,30,60,90,120,150,180}`, `parkinson_volatility_{10,20,30,60,90,120,150,180}`

</details>

---

## Project Structure

```
factor-alpha-platform/
├── src/
│   ├── operators/
│   │   ├── fastexpression.py    # BRAIN expression compiler (lexer → parser → AST → evaluator)
│   │   ├── vectorized.py        # 80+ vectorized DataFrame operators
│   │   ├── parser.py            # Per-date expression parser (legacy)
│   │   ├── cross_sectional.py   # Cross-sectional operator helpers
│   │   ├── element_wise.py      # Element-wise operator helpers
│   │   └── time_series.py       # Time-series operator helpers
│   ├── simulation/
│   │   ├── vectorized_sim.py    # Main vectorized backtester
│   │   ├── engine.py            # Simulation engine abstraction
│   │   ├── metrics.py           # Sharpe, fitness, drawdown calculations
│   │   ├── neutralization.py    # GICS group neutralization
│   │   └── oos.py               # Out-of-sample walk-forward testing
│   ├── data/
│   │   ├── bulk_download.py     # FMP data download pipeline
│   │   ├── fmp_loader.py        # Parquet matrix loader
│   │   ├── context_research.py  # InMemoryDataContext abstraction
│   │   ├── alpha_database.py    # SQLite alpha storage
│   │   ├── field_catalog.py     # WQ-compatible field definitions
│   │   └── synthetic.py         # Synthetic data generator for tests
│   ├── agent/
│   │   ├── gp_engine.py         # DEAP-based genetic programming
│   │   └── research_agent.py    # Gemini LLM alpha generation agent
│   ├── api/
│   │   └── server.py            # FastAPI REST + WebSocket server
│   ├── evaluation/
│   │   └── pipeline.py          # Alpha evaluation pipeline
│   ├── portfolio/
│   │   └── optimizer.py         # Multi-alpha portfolio optimization
│   └── core/
│       ├── types.py             # Pydantic data models
│       ├── config.py            # Configuration management
│       └── data_context.py      # Abstract data context interface
├── gui/
│   ├── index.html               # Alpha Library web interface
│   ├── app.js                   # Main application logic
│   ├── portfolio.js             # Portfolio construction UI
│   └── styles.css               # Styling
├── data/
│   └── fmp_cache/
│       ├── matrices/            # 121 Parquet data matrices
│       ├── universes/           # TOP200–TOP3000 universe masks
│       ├── classifications.json # GICS classification hierarchy
│       └── metadata.json        # Dataset metadata
├── tests/                       # Pytest test suite
├── build_gics_classifications.py    # GICS code builder (FMP → GICS mapping)
├── build_derived_fields.py          # Financial ratio computation
├── expand_universe.py               # Delisted stock downloader
├── run_gp_top1000.py                # GP alpha discovery runner
├── run_llm_sim.py                   # LLM simulation runner
├── test_wq_alpha.py                 # WQ alpha replication test
└── pyproject.toml                   # Project configuration
```

---

## License

MIT

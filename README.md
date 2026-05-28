# FINM3422 Assignment 3 - Risk and Derivative Platforms

A prototype risk and derivatives modelling platform simulating a trading/risk desk tool: it ingests market data, constructs a yield curve, prices equity derivatives, aggregates them into a portfolio, and computes portfolio risk (Greeks, VaR, and scenario analysis).

## Team Members
- Yi Chen (Cathy) Lin
- Caden Rieger
- Archer Hanrahan
- Sienna Hockaday

## Setup
Python 3.11 or later. Install runtime dependencies:
```bash
pip install numpy pandas matplotlib scipy
```
All market data is cached locally, so the notebooks run without any live data fetch. To **regenerate** the equity cache from source (optional), also install `yfinance`:
```bash
pip install yfinance
```

## Running the Notebooks
Open in Jupyter or VS Code and run all cells top to bottom. The recommended order follows the build of the platform from infrastructure up to integrated risk:

1. `notebooks/yield_curve_explanation.ipynb` – yield curve construction and visualisation
2. `notebooks/equity_data_explanation.ipynb` – equity data loading, returns, and volatility estimation
3. `notebooks/derivatives_pricing_test.ipynb` – option pricing and pricer cross-validation
4. `notebooks/trading_desk_analysis.ipynb` – **main notebook**: integrates all modules into portfolio valuation, Greeks, VaR, and scenario analysis

## Repository Structure
- `data/` — RBA F17 government bond yield data (`RBA_Government_Bond_Yields.csv`) and cached daily equity price data (`data/equities/`)
- `src/` — Reusable Python modules:
  - `yieldcurve.py` — loads RBA F17 data, interpolates zero rates, computes discount factors
  - `derivatives/` — `Derivative` base class and option subclasses (European, American, Barrier)
  - `pricers/` — Black-Scholes, binomial tree, and Monte Carlo pricing engines
  - `portfolio.py` — `Portfolio` and `RiskEngine`: valuation, Greeks, VaR, and scenarios
  - `equity_data.py` — equity panel loading, log returns, summary and correlation tables
- `notebooks/` — Analysis notebooks as described above

## Module Overview
The platform is layered. The yield curve is the infrastructure layer that all pricing depends on for discounting. The derivatives layer prices individual contracts, with each contract type matched to its natural pricer (Black-Scholes for European, binomial for American, Monte Carlo for the path-dependent barrier). The portfolio layer aggregates mixed equity and option positions into mark-to-market value, aggregate Greeks, three VaR methods, and full-revaluation scenario analysis.

## Reproducibility
Monte Carlo pricing uses a pinned random seed (`MC_SEED = 42`), so results are reproducible run to run. The yield curve and analytical pricers are deterministic given the same input data. All market data is cached locally as CSV — no live API calls are made at runtime — so results reproduce exactly from the bundled `data/` files.
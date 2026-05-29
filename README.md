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

## Methodology

### Yield Curve
The yield curve is constructed from RBA F17 daily nominal government bond data using linear interpolation between observed pillar maturities. Zero rates are converted to continuous discount factors. The curve is rebuilt from the most recent complete data snapshot each run.

### Equity Data
Five years of daily closing prices (May 2021 to May 2026) are loaded from local CSV cache for NAB.AX, BHP.AX, CSL.AX, and WOW.AX. Log returns are computed as `ln(P_t / P_{t-1})`. Annualised volatility is estimated as the sample standard deviation of daily log returns scaled by `sqrt(252)`.

### Derivative Pricing
Three independent pricing engines are implemented and cross-validated against each other:
- **Black-Scholes** - closed-form solution for European calls and puts under constant volatility and continuous discounting
- **Binomial tree** - Cox-Ross-Rubinstein lattice with N=200 steps, used for American options where early exercise premium is material
- **Monte Carlo** - GBM path simulation with 50,000 paths, used for path-dependent barrier options where closed-form solutions do not exist

Greeks (delta, gamma, vega, theta) are computed via central finite differences applied uniformly across all three pricers, ensuring model consistency regardless of contract type.

### Portfolio and Risk
Portfolio value is the sum of mark-to-market position values. Aggregate Greeks are computed by summing position-level Greeks, with equity positions contributing delta only (zero gamma, vega, theta).

Value-at-Risk is computed using three methods for cross-validation at 1-day and 10-day horizons:
- **Historical simulation** - empirical return quantile, no distributional assumption
- **Parametric (delta-normal)** - assumes normally distributed portfolio returns
- **Monte Carlo** - simulates portfolio returns from a fitted normal distribution

Bootstrap confidence intervals (1,000 resamples) are reported alongside historical VaR to quantify sampling uncertainty. Expected Shortfall (CVaR) is reported as the mean loss conditional on exceeding the VaR threshold.

Scenario analysis uses full revaluation - every position is repriced through its native engine under each shock - rather than delta approximation, so barrier activation and option non-linearity are captured correctly.

## Assumptions
- **Constant volatility** - each option is priced with a single historical sigma estimated from 5 years of daily returns. No volatility surface, smile, or skew is modelled.
- **No dividend yield** - all pricers assume a continuous dividend yield of zero (`q = 0`), which understates put values and overstates call values for dividend-paying underlyings. Most material for the NAB American put given NAB's trailing dividend yield of approximately 5%.
- **No transaction costs or market impact** - positions are valued at mid-market and all rebalancing is assumed costless.
- **Snapshot risk** - all Greeks, VaR, and scenario outputs are computed at a single point in time. The platform does not simulate dynamic re-hedging or path-dependent P&L.
- **Historical window** - the five-year sample (May 2021 to May 2026) sits after the March 2020 COVID drawdown, so the return distribution does not include that systemic stress event. VaR estimates should not be treated as worst-case bounds.

## Reproducibility
Monte Carlo pricing uses a pinned random seed (`MC_SEED = 42`), so results are reproducible run to run. The yield curve and analytical pricers are deterministic given the same input data. All market data is cached locally as CSV — no live API calls are made at runtime — so results reproduce exactly from the bundled `data/` files.Sonnet 4.6 Low
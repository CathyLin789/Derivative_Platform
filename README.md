# FINM3422 Assignment 3 - Risk and Derivative Platforms
## Team Members
- Yi Chen (Cathy) Lin
- Caden Rieger
- Archer Hanrahan
- Sienna Hockaday

## Setup
Python 3.11 or later. Install dependencies:
```bash
pip install numpy pandas matplotlib yfinance
```

## Running Notebooks
Open notebooks in the following order in Jupyter or VS Code and run all cells from top to bottom:
- `notebooks/yield_curve_explanation.ipynb` – yield curve construction and visualisation
- `notebooks/derivatives_pricing_test.ipynb` – option pricing and pricer cross-validation

## Reproducibility
Random seeds are pinned in the Monte Carlo simulation notebooks. The pricing engine and yield curve are deterministic given the same pillar data. To reproduce results exactly, use the bundled `data/f2-data.csv` rather than re-fetching live RBA data.

## Explanation of Notebooks

- `yield_curve_explanation.ipynb` – loads the RBA F2 government bond data, constructs a zero-rate yield curve via linear interpolation, and plots both the zero rate curve and discount factor curve
- `derivatives_pricing_test.ipynb` – prices European and American options using Black-Scholes, binomial tree, and Monte Carlo pricers; includes cross-validation and convergence checks

## Repository Structure
- `data/` — RBA F2 government bond yield data and cached equity price data
- `src/` — Reusable Python modules (yieldcurve.py, derivatives/, pricers/, portfolio.py)
- `notebooks/` — Analysis notebooks as described above
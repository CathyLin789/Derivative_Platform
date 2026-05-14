# Derivative_Platform
FINM3422 Assignment 3

A prototype risk and derivatives modelling platform built for a trading desk.

## Modules

- `src/yieldcurve.py` – Builds a zero-rate yield curve from live market data with interpolation, discount factors, and plotting
- `src/derivatives.py` – OOP-based derivatives pricing (Black-Scholes) with a base `Derivative` class and `EuropeanCall`/`EuropeanPut` subclasses

## Notebooks

- `notebooks/yield_curve_explanation.ipynb` – Yield curve construction and visualisation
- `notebooks/derivatives_pricing_test.ipynb` – Option pricing demonstrations

## Installation

```bash
pip install numpy pandas matplotlib yfinance
```

## How to Run

1. Clone the repo
2. Install dependencies above
3. Open and run `notebooks/yield_curve_explanation.ipynb` top to bottom

## Data Sources

Live US Treasury yields are fetched from Yahoo Finance via `yfinance`:
- `^IRX` – 13-week T-Bill (0.25yr)
- `^FVX` – 5-year Treasury Note
- `^TNX` – 10-year Treasury Note
- `^TYX` – 30-year Treasury Bond

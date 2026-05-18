"""
portfolio.py
------------
Portfolio construction and risk analysis layer.

Sits on top of derivatives.py, pricers.py, and yieldcurve.py to provide:
    - EquityPosition   : a long/short equity holding
    - OptionPosition   : a long/short derivative position with an attached pricer
    - Portfolio        : a collection of positions with aggregate valuation and delta
    - RiskEngine       : VaR (historical and parametric) and scenario analysis

Design principle (consistent with the rest of the platform):
    This module does NOT hard-code any specific stocks or options.
    Positions are constructed externally (e.g. in the notebook) and passed in.
    This keeps the risk engine reusable across different portfolio compositions.

Usage example (see trading_desk_analysis.ipynb for full demo):
    from src.derivatives import EuropeanCall, EuropeanPut, AsianCall
    from src.pricers import BlackScholesPricer, MonteCarloPricer
    from src.portfolio import EquityPosition, OptionPosition, Portfolio, RiskEngine
    from src.yieldcurve import load_rba_yield_curve, YieldCurve

    maturities, rates, snap = load_rba_yield_curve("data/f2-data.csv")
    yc = YieldCurve(maturities, rates)

    cba_equity = EquityPosition(ticker="CBA.AX", quantity=100, spot_price=138.50)
    cba_put    = OptionPosition(
                     contract=EuropeanPut(S0=138.50, K=131.57, T=0.5, sigma=0.18, yield_curve=yc),
                     pricer=BlackScholesPricer(),
                     quantity=2,
                     label="CBA.AX Put 6m OTM"
                 )
    portfolio  = Portfolio([cba_equity, cba_put])
    engine     = RiskEngine(portfolio, returns_df)
    print(engine.historical_var())
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ======================================================================
# Position classes
# ======================================================================

class EquityPosition:
    """
    A long equity holding in a single stock.

    Parameters
    ----------
    ticker : str
        ASX ticker (e.g. "CBA.AX"). Used for labelling and linking to
        return data in the RiskEngine.
    quantity : float
        Number of shares held. Positive = long.
    spot_price : float
        Current market price per share (AUD).

    Notes
    -----
    Delta of an equity position is always 1.0 per share, so the
    position delta equals quantity.
    """

    def __init__(self, ticker, quantity, spot_price):
        if spot_price <= 0:
            raise ValueError(f"spot_price must be positive, got {spot_price}")
        self.ticker = ticker
        self.quantity = float(quantity)
        self.spot_price = float(spot_price)
        self.label = ticker

    def value(self):
        """Total market value of the equity position (AUD)."""
        return self.quantity * self.spot_price

    def delta(self):
        """
        Position delta.
        Delta per share = 1.0 for a long equity holding.
        """
        return self.quantity * 1.0

    def reprice(self, new_spot=None, new_yield_curve=None):
        """
        Return a new EquityPosition with shocked inputs.

        Parameters
        ----------
        new_spot : float or None
            Shocked spot price. If None, keeps current spot_price.
        new_yield_curve : ignored
            Included for API consistency with OptionPosition.reprice().
            Equity value does not depend on the yield curve directly.

        Returns
        -------
        EquityPosition
        """
        spot = new_spot if new_spot is not None else self.spot_price
        return EquityPosition(
            ticker=self.ticker,
            quantity=self.quantity,
            spot_price=spot,
        )

    def __repr__(self):
        return (
            f"EquityPosition(ticker={self.ticker!r}, "
            f"quantity={self.quantity}, spot_price={self.spot_price:.2f})"
        )


class OptionPosition:
    """
    A long position in a single derivative contract.

    Parameters
    ----------
    contract : Derivative
        A derivative contract (EuropeanCall, EuropeanPut, AsianCall, etc.)
        as defined in src/derivatives.py.
    pricer : Pricer
        A compatible pricer (BlackScholesPricer, MonteCarloPricer, etc.)
        as defined in src/pricers.py.
    quantity : float
        Number of contracts held. Positive = long.
    label : str, optional
        Human-readable label for reporting (e.g. "CBA.AX Put 6m OTM").
        Convention: start with the ticker so RiskEngine can infer the
        underlying (e.g. "CBA.AX Put 6m OTM", "BHP.AX Asian Call 3m ATM").

    Notes
    -----
    Delta is computed numerically via a central finite difference on the
    spot price. This is model-consistent and works for any pricer, including
    Monte Carlo (though MC delta will have simulation noise).

    For Black-Scholes European options the analytical delta is:
        Call: N(d1)       Put: N(d1) - 1
    The finite-difference result converges to these values as dS -> 0.
    """

    # Relative bump size for finite-difference delta (0.1% of spot)
    _DELTA_BUMP = 1e-3

    def __init__(self, contract, pricer, quantity, label=None):
        self.contract = contract
        self.pricer = pricer
        self.quantity = float(quantity)
        self.label = label or repr(contract)

    def value(self):
        """Total value of the option position (AUD)."""
        return self.quantity * self.pricer.price(self.contract)

    def delta(self):
        """
        Position delta, computed via central finite difference on S0.

        delta_per_contract = (price(S0 + dS) - price(S0 - dS)) / (2 * dS)
        position_delta     = quantity * delta_per_contract
        """
        S0 = self.contract.S0
        dS = S0 * self._DELTA_BUMP

        contract_up = _clone_contract_with(self.contract, S0=S0 + dS)
        contract_dn = _clone_contract_with(self.contract, S0=S0 - dS)

        price_up = self.pricer.price(contract_up)
        price_dn = self.pricer.price(contract_dn)

        delta_per_contract = (price_up - price_dn) / (2.0 * dS)
        return self.quantity * delta_per_contract

    def reprice(self, new_spot=None, new_yield_curve=None):
        """
        Return a new OptionPosition with shocked inputs.

        Parameters
        ----------
        new_spot : float or None
            Shocked spot price (S0). If None, keeps current S0.
        new_yield_curve : YieldCurve or None
            Shocked yield curve. If None, keeps current yield curve.

        Returns
        -------
        OptionPosition
            A new position with the same quantity, pricer, and label,
            but with a contract built from the shocked inputs.
        """
        new_S0 = new_spot if new_spot is not None else self.contract.S0
        new_yc = new_yield_curve if new_yield_curve is not None else self.contract.yield_curve

        shocked_contract = _clone_contract_with(
            self.contract, S0=new_S0, yield_curve=new_yc
        )
        return OptionPosition(
            contract=shocked_contract,
            pricer=self.pricer,
            quantity=self.quantity,
            label=self.label,
        )

    def __repr__(self):
        return (
            f"OptionPosition(label={self.label!r}, "
            f"quantity={self.quantity}, pricer={self.pricer.name!r})"
        )


# ======================================================================
# Portfolio
# ======================================================================

class Portfolio:
    """
    A collection of equity and option positions.

    Parameters
    ----------
    positions : list
        List of EquityPosition and/or OptionPosition objects.

    Methods
    -------
    total_value()       -> float         : sum of all position values
    portfolio_delta()   -> float         : sum of all position deltas
    summary_table()     -> pd.DataFrame  : per-position breakdown
    """

    def __init__(self, positions):
        if not positions:
            raise ValueError("Portfolio must contain at least one position.")
        self.positions = positions

    def total_value(self):
        """Aggregate mark-to-market value of the portfolio (AUD)."""
        return sum(p.value() for p in self.positions)

    def portfolio_delta(self):
        """
        Aggregate delta of the portfolio.

        Interpretation: approximate change in portfolio value for a $1
        increase in the underlying price. For a multi-stock portfolio this
        is a simplification — it assumes all underlyings move together.
        """
        return sum(p.delta() for p in self.positions)

    def summary_table(self):
        """
        Return a DataFrame with one row per position showing:
            Label, Type, Quantity, Unit Value, Position Value, Delta

        Useful for display in the notebook.
        """
        rows = []
        for p in self.positions:
            if isinstance(p, EquityPosition):
                position_type = "Equity"
                unit_value = p.spot_price
            else:
                position_type = type(p.contract).__name__
                unit_value = p.pricer.price(p.contract)

            rows.append({
                "Label":              p.label,
                "Type":               position_type,
                "Quantity":           p.quantity,
                "Unit Value ($)":     round(unit_value, 4),
                "Position Value ($)": round(p.value(), 2),
                "Delta":              round(p.delta(), 4),
            })

        df = pd.DataFrame(rows)
        df.loc["Total"] = {
            "Label":              "TOTAL",
            "Type":               "",
            "Quantity":           "",
            "Unit Value ($)":     "",
            "Position Value ($)": round(self.total_value(), 2),
            "Delta":              round(self.portfolio_delta(), 4),
        }
        return df

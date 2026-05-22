"""
portfolio.py
------------
Portfolio construction and risk analysis layer.

Sits on top of derivatives/, pricers/, and yieldcurve.py to provide:
    - EquityPosition   : a long/short equity holding
    - OptionPosition   : a long/short derivative position with an attached pricer
    - Portfolio        : a collection of positions with aggregate valuation and delta
    - RiskEngine       : VaR (historical and parametric) and scenario analysis

Design principle (consistent with the rest of the platform):
    This module does NOT hard-code any specific stocks or options.
    Positions are constructed externally (e.g. in the notebook) and passed in.
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
    """

    def __init__(self, ticker, quantity, spot_price):
        if spot_price <= 0:
            raise ValueError(f"spot_price must be positive, got {spot_price}")
        self.ticker     = ticker
        self.quantity   = float(quantity)
        self.spot_price = float(spot_price)
        self.label      = ticker

    def price(self):
        """Unit price (spot)."""
        return self.spot_price
        
    def value(self):
        """Total market value of the equity position (AUD)."""
        return self.quantity * self.spot_price

    def delta(self):
        """ Unit delta = 1.0 for equity. Position delta = quantity."""
        return 1.0

    def reprice(self, new_spot=None, new_yield_curve=None):
        """Return a new EquityPosition with shocked spot price."""
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
    A long derivative position with an attached pricer.
 
    Delta is computed via central finite difference on S0 — model-consistent
    and works for any pricer including Monte Carlo.
    """
    # Relative bump size for finite-difference delta (0.1% of spot)
    _DELTA_BUMP = 1e-3

    def __init__(self, contract, pricer, quantity, underlying_ticker, label=None):
        self.contract           = contract
        self.pricer             = pricer
        self.quantity           = float(quantity)
        self.underlying_ticker  = underlying_ticker
        self.label              = label or repr(contract)
    
    def price(self):
        """Unit price (single contract)."""
        return self.pricer.price(self.contract)

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
        price_up = self.pricer.price(_clone_contract_with(self.contract, S0=S0 + dS))
        price_dn = self.pricer.price(_clone_contract_with(self.contract, S0=S0 - dS))
        
        return (price_up - price_dn) / (2.0 * dS)

    def reprice(self, new_spot=None, new_yield_curve=None):
        """Return a new OptionPosition with shocked inputs."""
        new_S0 = new_spot if new_spot is not None else self.contract.S0
        new_yc = new_yield_curve if new_yield_curve is not None else self.contract.yield_curve

        return OptionPosition(
            contract=_clone_contract_with(self.contract, S0=new_S0, yield_curve=new_yc),
            pricer=self.pricer,
            quantity=self.quantity,
            underlying_ticker=self.underlying_ticker,
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
        value()          -> float  : total portfolio value
        delta()          -> float  : total portfolio delta
        add_position()             : add a position
        position_table() -> DataFrame : per-position breakdown (tutorial name)
        summary_table()  -> DataFrame : alias for position_table()
    """

    def __init__(self, positions=None):
        self.positions = []
        if positions:
            for p in positions:
                self.add_position(p, quantity=p.quantity,
                                  label=getattr(p, 'label', None))
 
    def add_position(self, instrument, quantity=None, label=None):
        """
        Add a position
        """
        # If instrument is already a position object, quantity lives on it
        if quantity is None:
            quantity = getattr(instrument, 'quantity', 1.0)
        self.positions.append({
            "instrument": instrument,
            "quantity":   float(quantity),
            "label":      label or getattr(instrument, 'label', None),
        })

    def value(self):
        """Compute total portfolio value."""
        total_value = 0.0
        for position in self.positions:
            instrument = position["instrument"]
            quantity   = position["quantity"]
            total_value += quantity * instrument.price()
        return total_value

    def total_value(self):
        """Aggregate mark-to-market value of the portfolio (AUD)."""
        return self.value()

    def delta(self):
        """Compute total portfolio delta."""
        total_delta = 0.0
        for position in self.positions:
            instrument = position["instrument"]
            quantity   = position["quantity"]
            total_delta += quantity * instrument.delta()
        return total_delta

    def portfolio_delta(self):
        return self.delta()


    def position_table(self):
        """
        Per-position breakdown matching tutorial column names exactly:
        Position, Quantity, Unit Value, Position Value, Unit Delta, Position Delta
        """
        rows = []
        for position in self.positions:
            instrument = position["instrument"]
            quantity   = position["quantity"]
 
            if position["label"] is not None:
                name = position["label"]
            elif hasattr(instrument, "ticker"):
                name = instrument.ticker
            else:
                name = instrument.__class__.__name__
 
            unit_value = instrument.price()
            unit_delta = instrument.delta()

            rows.append({
                "Position":       name,
                "Quantity":       quantity,
                "Unit Value":     unit_value,
                "Position Value": quantity * unit_value,
                "Unit Delta":     unit_delta,
                "Position Delta": quantity * unit_delta,
            })

        df = pd.DataFrame(rows)
        if len(df) > 0:
            total_row = pd.DataFrame([{
                "Position":       "TOTAL",
                "Quantity":       np.nan,
                "Unit Value":     np.nan,
                "Position Value": df["Position Value"].sum(),
                "Unit Delta":     np.nan,
                "Position Delta": df["Position Delta"].sum(),
            }])
            df = pd.concat([df, total_row], ignore_index=True)
 
        return df
    
    def summary_table(self):
        """Alias for position_table() — used by the trading desk notebook."""
        return self.position_table()


# ======================================================================
# Risk Engine
# ======================================================================

class RiskEngine:
    """
    Portfolio risk metrics: historical VaR, parametric VaR, scenario analysis.
 
    historical_var(returns, alpha, horizon_days)
    but also accepts no arguments (uses stored returns_df).
 
    Extension beyond tutorial:
        - parametric_var()
        - scenario_analysis() / scenario_pnl_table()
        - _portfolio_returns() for multi-ticker delta-normal weighting
    """

    def __init__(self, portfolio, returns_df):
        self.portfolio = portfolio
        self.returns_df = returns_df.copy()

    # ------------------------------------------------------------------
    # VaR methods
    # ------------------------------------------------------------------

    def historical_var(self, confidence=0.95, horizon=1):
        """
        Historical simulation VaR.
        Returns VaR in dollars at the given confidence level and horizon.
        """
        port_returns = self._portfolio_returns()
        scaled_returns = port_returns * np.sqrt(horizon)
        portfolio_value = self.portfolio.total_value()
        var = -np.quantile(scaled_returns, 1 - confidence) * abs(portfolio_value)
        return float(var)

    def parametric_var(self, confidence=0.95, horizon=1):
        """
        Parametric (delta-normal) VaR. Assumes normally distributed returns.
        """
        port_returns = self._portfolio_returns()
        sigma = port_returns.std()
        z = norm.ppf(confidence)
        portfolio_value = self.portfolio.total_value()
        var = z * sigma * np.sqrt(horizon) * abs(portfolio_value)
        return float(var)

    def var_summary(self, confidence=0.95, horizon=1):
        """DataFrame comparing historical and parametric VaR."""
        hist     = self.historical_var(confidence, horizon)
        para     = self.parametric_var(confidence, horizon)
        port_val = self.portfolio.value()
        return pd.DataFrame({
            "Method":         ["Historical", "Parametric"],
            "Confidence":     [f"{confidence:.0%}", f"{confidence:.0%}"],
            "Horizon (days)": [horizon, horizon],
            "VaR ($)":        [round(hist, 2), round(para, 2)],
            "VaR (% NAV)":    [f"{hist/port_val:.2%}", f"{para/port_val:.2%}"],
        })

    # ------------------------------------------------------------------
    # Scenario analysis
    # ------------------------------------------------------------------

    def scenario_analysis(self, spot_shocks=None, rate_shocks=None):
        """
        Reprice under a grid of spot and rate shocks.
        Returns DataFrame of P&L relative to base value.
        """
        if spot_shocks is None:
            spot_shocks = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
        if rate_shocks is None:
            rate_shocks = [-0.01, -0.005, 0.0, 0.005, 0.01]

        base_value = self.portfolio.total_value()
        results = {}

        for rate_shock in rate_shocks:
            col_pnl = []
            for spot_shock in spot_shocks:
                shocked_value = self._reprice_portfolio(spot_shock, rate_shock)
                pnl = shocked_value - base_value
                col_pnl.append(round(pnl, 2))
            col_label = f"Rate {rate_shock * 10000:+.0f}bps"
            results[col_label] = col_pnl

        index_labels = [f"Spot {s * 100:+.0f}%" for s in spot_shocks]
        return pd.DataFrame(results, index=index_labels)

    def scenario_pnl_table(self, spot_shocks=None, rate_shocks=None):
        """
        Return both a dollar P&L table and a percentage P&L table.
        Useful for displaying side-by-side in the notebook.

        Returns
        -------
        dollar_pnl : pd.DataFrame
        pct_pnl    : pd.DataFrame
        """
        base_value = self.portfolio.total_value()
        dollar_pnl = self.scenario_analysis(spot_shocks, rate_shocks)
        pct_pnl = (dollar_pnl / abs(base_value) * 100).round(2)
        pct_pnl = pct_pnl.map(lambda x: f"{x:+.2f}%")
        return dollar_pnl, pct_pnl

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _portfolio_returns(self):
        total_value  = self.portfolio.value()
        port_returns = pd.Series(np.zeros(len(self.returns_df)),
                                index=self.returns_df.index)

        for position in self.portfolio.positions:
            instrument = position["instrument"]
            quantity   = position["quantity"]

            if isinstance(instrument, EquityPosition):
                ticker = instrument.ticker
                if ticker not in self.returns_df.columns:
                    raise ValueError(f"Ticker {ticker!r} not in returns_df. "
                                    f"Available: {list(self.returns_df.columns)}")
                weight = (quantity * instrument.price()) / total_value
                port_returns += weight * self.returns_df[ticker]

            elif isinstance(instrument, OptionPosition):
                ticker = instrument.underlying_ticker
                if ticker not in self.returns_df.columns:
                    raise ValueError(f"Ticker {ticker!r} not in returns_df. "
                                    f"Available: {list(self.returns_df.columns)}")
                dollar_delta = quantity * instrument.delta() * instrument.contract.S0
                weight       = dollar_delta / total_value
                port_returns += weight * self.returns_df[ticker]

        return port_returns

    def _reprice_portfolio(self, spot_shock, rate_shock):
        total = 0.0
        for position in self.portfolio.positions:
            instrument = position["instrument"]
            quantity   = position["quantity"]

            if isinstance(instrument, EquityPosition):
                new_spot = instrument.spot_price * (1.0 + spot_shock)
                total   += quantity * instrument.reprice(new_spot=new_spot).price()

            elif isinstance(instrument, OptionPosition):
                new_spot = instrument.contract.S0 * (1.0 + spot_shock)
                new_yc   = _shift_yield_curve(instrument.contract.yield_curve, rate_shock)
                total   += quantity * instrument.reprice(new_spot=new_spot,
                                                        new_yield_curve=new_yc).price()
        return total


# ======================================================================
# Internal utility functions
# ======================================================================

def _clone_contract_with(contract, **overrides):
    """
    Return a new contract of the same type as `contract`, with any
    parameters in `overrides` replaced.

    Used by OptionPosition.delta() and OptionPosition.reprice() to create
    shocked versions of a contract without modifying the original.

    Carries forward base Derivative parameters plus any contract-specific
    parameters (e.g. barrier, barrier_type for BarrierCall/BarrierPut)
    so that subclasses with extended __init__ signatures clone correctly.
    """
    # Base parameters every Derivative subclass accepts
    params = {
        "S0":          contract.S0,
        "K":           contract.K,
        "T":           contract.T,
        "sigma":       contract.sigma,
        "yield_curve": contract.yield_curve,
    }
    # Barrier-specific parameters, present only on BarrierCall/BarrierPut
    if hasattr(contract, "barrier"):
        params["barrier"]      = contract.barrier
        params["barrier_type"] = contract.barrier_type
    params.update(overrides)
    return type(contract)(**params)


def _shift_yield_curve(yield_curve, shift):
    """New YieldCurve with all zero rates shifted by `shift` (decimal)."""
    from src.yieldcurve import YieldCurve
    shifted_rates = yield_curve.zero_rates + shift
    return YieldCurve(
        maturities=yield_curve.maturities.copy(),
        zero_rates=shifted_rates,
        compounding=yield_curve.compounding,
    )
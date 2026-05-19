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
    
# ======================================================================
# Risk Engine
# ======================================================================

class RiskEngine:
    """
    Computes portfolio risk metrics: VaR and scenario analysis.

    Parameters
    ----------
    portfolio : Portfolio
        The portfolio to analyse.
    returns_df : pd.DataFrame
        Daily log returns for the equity underlyings. Each column is a
        ticker (e.g. "CBA.AX"), each row is a trading day.
        Used for historical VaR and parametric volatility estimation.

    Notes
    -----
    VaR is computed at the portfolio level using a delta-normal approximation:
        - Equity positions contribute directly via their dollar weight.
        - Option positions contribute via delta * S0 as their dollar equity
          exposure (linear approximation).
        - This does not capture the nonlinear (gamma) effect of options on P&L.

    Full revaluation under each historical scenario would be more accurate
    but computationally prohibitive for Monte Carlo-priced options. This
    limitation is acknowledged in the presentation as a known simplification
    consistent with a prototype trading desk system.
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

        Uses the empirical distribution of past portfolio P&L to find
        the loss threshold at the given confidence level.

        Procedure
        ---------
        1. Compute daily portfolio P&L from historical equity returns,
           weighted by each position's dollar exposure.
        2. Scale to the desired horizon (sqrt-of-time approximation).
        3. Return the (1 - confidence) quantile of the loss distribution.

        Parameters
        ----------
        confidence : float
            Confidence level. Default 0.95 -> 95% VaR.
        horizon : int
            Holding period in days. Default 1 (1-day VaR).

        Returns
        -------
        float
            VaR in dollars. Positive number = potential loss.
        """
        port_returns = self._portfolio_returns()
        scaled_returns = port_returns * np.sqrt(horizon)
        var = -np.quantile(scaled_returns, 1 - confidence)
        return float(var)

    def parametric_var(self, confidence=0.95, horizon=1):
        """
        Parametric (variance-covariance) VaR.

        Assumes portfolio returns are normally distributed. Uses the
        historical standard deviation of portfolio returns scaled to
        the desired horizon.

        Procedure
        ---------
        1. Compute daily portfolio returns (same as historical VaR).
        2. Estimate portfolio volatility as the standard deviation.
        3. VaR = z_alpha * sigma * sqrt(horizon) * abs(portfolio_value)

        Parameters
        ----------
        confidence : float
            Confidence level. Default 0.95 -> z = 1.645.
        horizon : int
            Holding period in days. Default 1.

        Returns
        -------
        float
            VaR in dollars. Positive number = potential loss.
        """
        port_returns = self._portfolio_returns()
        sigma = port_returns.std()
        z = norm.ppf(confidence)
        portfolio_value = self.portfolio.total_value()
        var = z * sigma * np.sqrt(horizon) * abs(portfolio_value)
        return float(var)

    def var_summary(self, confidence=0.95, horizon=1):
        """
        Return a DataFrame comparing historical and parametric VaR.

        Useful for displaying in the notebook and discussing the
        normality assumption as a limitation.
        """
        hist = self.historical_var(confidence, horizon)
        para = self.parametric_var(confidence, horizon)
        portfolio_value = self.portfolio.total_value()

        return pd.DataFrame({
            "Method":         ["Historical", "Parametric"],
            "Confidence":     [f"{confidence:.0%}", f"{confidence:.0%}"],
            "Horizon (days)": [horizon, horizon],
            "VaR ($)":        [round(hist, 2), round(para, 2)],
            "VaR (% NAV)":    [
                f"{hist / portfolio_value:.2%}",
                f"{para / portfolio_value:.2%}",
            ],
        })

    # ------------------------------------------------------------------
    # Scenario analysis
    # ------------------------------------------------------------------

    def scenario_analysis(self, spot_shocks=None, rate_shocks=None):
        """
        Reprice the portfolio under a grid of spot and rate shocks.

        For each combination of shocks, every position is repriced:
            - EquityPosition : new_spot = current_spot * (1 + spot_shock)
            - OptionPosition : new S0 and/or new yield curve (parallel shift)

        Parameters
        ----------
        spot_shocks : list of float, optional
            Fractional spot price changes.
            Default: [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]
        rate_shocks : list of float, optional
            Parallel yield curve shifts in decimal.
            Default: [-0.01, -0.005, 0.0, 0.005, 0.01]  (±50bps, ±100bps)

        Returns
        -------
        pd.DataFrame
            Rows = spot shocks, Columns = rate shocks.
            Each cell = portfolio P&L relative to base value (AUD).
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
        """
        Compute daily portfolio-level returns as a weighted sum of
        equity returns, where weights are proportional to dollar exposure.

        Equity positions: weight = position value / total portfolio value
        Option positions: weight = (delta * S0) / total portfolio value
                          (delta-normal approximation)
        """
        total_value = self.portfolio.total_value()
        port_returns = pd.Series(
            np.zeros(len(self.returns_df)), index=self.returns_df.index
        )

        for p in self.portfolio.positions:
            if isinstance(p, EquityPosition):
                ticker = p.ticker
                if ticker not in self.returns_df.columns:
                    raise ValueError(
                        f"Ticker {ticker!r} not found in returns_df. "
                        f"Available: {list(self.returns_df.columns)}"
                    )
                weight = p.value() / total_value
                port_returns += weight * self.returns_df[ticker]

            elif isinstance(p, OptionPosition):
                underlying_ticker = _infer_ticker(p)
                if underlying_ticker and underlying_ticker in self.returns_df.columns:
                    dollar_delta = p.delta() * p.contract.S0
                    weight = dollar_delta / total_value
                    port_returns += weight * self.returns_df[underlying_ticker]

        return port_returns

    def _reprice_portfolio(self, spot_shock, rate_shock):
        """
        Return total portfolio value after applying spot and rate shocks.

        Spot shock applied proportionally to each position's current S0.
        Rate shock is a parallel shift applied to the yield curve.
        """
        total = 0.0
        for p in self.portfolio.positions:
            if isinstance(p, EquityPosition):
                new_spot = p.spot_price * (1.0 + spot_shock)
                shocked = p.reprice(new_spot=new_spot)
                total += shocked.value()

            elif isinstance(p, OptionPosition):
                new_spot = p.contract.S0 * (1.0 + spot_shock)
                new_yc = _shift_yield_curve(p.contract.yield_curve, rate_shock)
                shocked = p.reprice(new_spot=new_spot, new_yield_curve=new_yc)
                total += shocked.value()

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
    """
    params = {
        "S0":          contract.S0,
        "K":           contract.K,
        "T":           contract.T,
        "sigma":       contract.sigma,
        "yield_curve": contract.yield_curve,
        "q":           contract.q,
    }
    params.update(overrides)
    return type(contract)(**params)


def _shift_yield_curve(yield_curve, shift):
    """
    Return a new YieldCurve with all zero rates shifted by `shift` (decimal).

    Used for parallel rate shock scenarios, e.g. shift=0.005 -> +50bps.
    """
    from src.yieldcurve import YieldCurve
    shifted_rates = yield_curve.zero_rates + shift
    return YieldCurve(
        maturities=yield_curve.maturities.copy(),
        zero_rates=shifted_rates,
        compounding=yield_curve.compounding,
    )


def _infer_ticker(option_position):
    """
    Extract the underlying ticker from an OptionPosition's label.

    Convention: labels start with the ticker, e.g. "CBA.AX Put 6m OTM".
    Returns None if the ticker cannot be inferred.
    """
    label = option_position.label or ""
    tokens = label.split()
    if tokens and tokens[0].endswith(".AX"):
        return tokens[0]
    return None